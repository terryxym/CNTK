//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <limits>
#include "MLFDataDeserializer.h"
#include "ConfigHelper.h"
#include "SequenceData.h"
#include "../HTKMLFReader/htkfeatio.h"
#include "../HTKMLFReader/msra_mgram.h"
#include "latticearchive.h"
#include "StringUtil.h"
#include "MLFIndexer.h"
#include "MLFUtils.h"

#undef max // max is defined in minwindef.h

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

static float s_oneFloat = 1.0;
static double s_oneDouble = 1.0;
static const double htkTimeToFrame = 100000.0; // default is 10ms

// Base chunk for frame and sequence mode.
class MLFDataDeserializer::ChunkBase : public Chunk
{
protected:
    std::vector<char> m_buffer;
    MLFUtteranceParser m_parser;

    const MLFDataDeserializer& m_parent;
    const ChunkDescriptor& m_descriptor;

public:
    ChunkBase(const MLFDataDeserializer& parent, const ChunkDescriptor& descriptor, const std::wstring& fileName, std::shared_ptr<FILE>& f, StateTablePtr states)
        : m_parser(states),
          m_descriptor(descriptor),
          m_parent(parent)
    {
        // Let's see if the open descriptor has problems.
        if (ferror(f.get()) != 0)
            f.reset(fopenOrDie(fileName.c_str(), L"rbS"), [](FILE* f) { if (f) fclose(f); });

        if (descriptor.m_sequences.empty() || !descriptor.m_byteSize)
            LogicError("Empty chunks are not supported.");

        m_buffer.resize(descriptor.m_byteSize + 1);

        // Make sure we always have 0 at the end for buffer overrun.
        m_buffer[descriptor.m_byteSize] = 0;

        auto chunkOffset = descriptor.m_sequences.front().m_fileOffsetBytes;

        // Seek and read chunk into memory.
        int rc = _fseeki64(f.get(), chunkOffset, SEEK_SET);
        if (rc)
            RuntimeError("Error seeking to position '%" PRId64 "' in the input file '%ls', error code '%d'", chunkOffset, fileName.c_str(), rc);

        freadOrDie(m_buffer.data(), descriptor.m_byteSize, 1, f.get());
    }
};

// Sequence MLF chunk. The time of life always less than the time of life of the parent deserializer.
class MLFDataDeserializer::SequenceChunk : public MLFDataDeserializer::ChunkBase
{
public:
    SequenceChunk(const MLFDataDeserializer& parent, const ChunkDescriptor& descriptor, const std::wstring& fileName, std::shared_ptr<FILE>& f, StateTablePtr states)
        : ChunkBase(parent, descriptor, fileName, f, states)
    {
    }

    void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override
    {
        const auto& sequence = m_descriptor.m_sequences[sequenceIndex];
        auto start = m_buffer.data() + (sequence.m_fileOffsetBytes - m_descriptor.m_sequences.front().m_fileOffsetBytes);
        auto end = start + sequence.m_byteSize;

        std::vector<MLFFrameRange> utterance;
        bool parsed = m_parser.Parse(sequence, boost::make_iterator_range(start, end), utterance, htkTimeToFrame);
        if (!parsed) // cannot parse
        {
            fprintf(stderr, "WARNING: Cannot parse the utterance %s", m_parent.m_corpus->IdToKey(sequence.m_key.m_sequence).c_str());
            SparseSequenceDataPtr s = make_shared<MLFSequenceData<float>>(0);
            s->m_isValid = false;
            result.push_back(s);
            return;
        }

        // Compute some statistics and perform checks.
        vector<size_t> sequencePhoneBoundaries(m_parent.m_withPhoneBoundaries ? utterance.size() : 0);
        size_t numSamples = 0;
        for (size_t i = 0; i < utterance.size(); ++i)
        {
            if (m_parent.m_withPhoneBoundaries)
                sequencePhoneBoundaries[i] = utterance[i].FirstFrame();

            const auto& range = utterance[i];
            if (range.ClassId() >= m_parent.m_dimension)
                RuntimeError("Class id %d exceeds the model output dimension %d.", (int)range.ClassId, (int)m_parent.m_dimension);

            numSamples += range.NumFrames();
        }

        // Packing labels for the utterance into sparse sequence.
        SparseSequenceDataPtr s;
        if (m_parent.m_elementType == ElementType::tfloat)
            s = make_shared<MLFSequenceData<float>>(numSamples, m_parent.m_withPhoneBoundaries ? sequencePhoneBoundaries : vector<size_t>{});
        else
        {
            assert(m_elementType == ElementType::tdouble);
            s = make_shared<MLFSequenceData<double>>(numSamples, m_parent.m_withPhoneBoundaries ? sequencePhoneBoundaries : vector<size_t>{});
        }

        {
            size_t frameIndex = 0;
            for (const auto& f : utterance)
            {
                for (size_t j = 0; j < f.NumFrames(); ++j)
                {
                    s->m_indices[frameIndex++] = static_cast<IndexType>(f.ClassId());
                }
            }
        }

        result.push_back(s);
    }
};



// MLF chunk. The time of life always less than the time of life of the parent deserializer.
class MLFDataDeserializer::FrameChunk : public MLFDataDeserializer::ChunkBase
{
    // Actual values of frames.
    std::vector<ClassIdType> m_classIds;

    // Index of the first frame of each and evey utterance.
    std::vector<size_t> m_firstFrames;

    // Mask whether the sequence was cached.
    std::vector<bool> m_cached;

    std::set<size_t> m_invalidUtterances;

    // Actually needed only in frame mode.
    std::mutex m_cacheLock;

public:
    FrameChunk(const MLFDataDeserializer& parent, const ChunkDescriptor& descriptor, const std::wstring& fileName, std::shared_ptr<FILE>& f, StateTablePtr states)
        : ChunkBase(parent, descriptor, fileName, f, states)
    {
        // Let's also preallocate an big array for filling in class ids for whole chunk,
        // it is used for optimizing speed of retrieval in frame mode.
        m_classIds.resize(m_descriptor.m_numberOfSamples);

        // Also prefil information where frames of a particular sequence start.
        {
            m_firstFrames.resize(m_descriptor.m_numberOfSequences);
            m_cached.resize(m_descriptor.m_numberOfSequences);
            size_t totalNumOfFrames = 0;
            for (size_t i = 0; i < m_descriptor.m_sequences.size(); ++i)
            {
                m_firstFrames[i] = totalNumOfFrames;
                totalNumOfFrames += m_descriptor.m_sequences[i].m_numberOfSamples;
            }
        }
    }

    // Get utterance by the absolute frame index in chunk.
    // Uses the upper bound to do the binary search among sequences of the chunk.
    size_t GetUtteranceForChunkFrameIndex(size_t frameIndex) const
    {
        auto result = std::upper_bound(
            m_firstFrames.begin(),
            m_firstFrames.end(),
            frameIndex,
            [](size_t fi, const size_t& a) { return fi < a; });
        return result - 1 - m_firstFrames.begin();
    }

    void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override
    {
        size_t utteranceId = GetUtteranceForChunkFrameIndex(sequenceIndex);
        CacheSequence(utteranceId);

        {
            std::lock_guard<std::mutex> lock(m_cacheLock);
            if (m_invalidUtterances.find(utteranceId) != m_invalidUtterances.end())
            {
                SparseSequenceDataPtr s = make_shared<MLFSequenceData<float>>(0);;
                s->m_isValid = false;
                result.push_back(s);
                return;
            }
        }

        size_t label = m_classIds[sequenceIndex];
        assert(label < m_parent.m_categories.size());
        result.push_back(m_parent.m_categories[label]);
    }

    // Parses and caches sequence in the buffer for future fast retrieval in frame mode.
    void CacheSequence(size_t sequenceIndex)
    {
        {
            std::lock_guard<std::mutex> lock(m_cacheLock);
            if (m_cached[sequenceIndex])
                return;
        }

        const auto& sequence = m_descriptor.m_sequences[sequenceIndex];
        auto start = m_buffer.data() + (sequence.m_fileOffsetBytes - m_descriptor.m_sequences.front().m_fileOffsetBytes);
        auto end = start + sequence.m_byteSize;

        std::vector<MLFFrameRange> utterance;
        bool parsed = m_parser.Parse(sequence, boost::make_iterator_range(start, end), utterance, htkTimeToFrame);
        if (!parsed)
        {
            std::lock_guard<std::mutex> lock(m_cacheLock);
            fprintf(stderr, "WARNING: Cannot parse the utterance %s", m_parent.m_corpus->IdToKey(sequence.m_key.m_sequence).c_str());
            m_invalidUtterances.insert(sequenceIndex);
            m_cached[sequenceIndex] = true;
            return;
        }

        std::vector<ClassIdType> localBuffer;
        localBuffer.resize(sequence.m_numberOfSamples);

        size_t total = 0;
        for(size_t i = 0; i < utterance.size(); ++i)
        {
            const auto& range = utterance[i];
            if (range.ClassId() >= m_parent.m_dimension)
                RuntimeError("Class id %d exceeds the model output dimension %d.", (int)range.ClassId(), (int)m_parent.m_dimension);

            memset(localBuffer.data() + total, utterance[i].NumFrames(), range.ClassId());
            total += utterance[i].NumFrames();
        }

        {
            std::lock_guard<std::mutex> lock(m_cacheLock);
            if (!m_cached[sequenceIndex])
            {
                memcpy(m_classIds.data() + m_firstFrames[sequenceIndex], localBuffer.data(), total);
                m_cached[sequenceIndex] = true;
            }
        }
    }
};

// Inner class for an utterance.
struct MLFUtterance : SequenceDescription
{
    size_t m_sequenceStart;
};

MLFDataDeserializer::MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& cfg, bool primary)
    : DataDeserializerBase(primary)
{
    // TODO: This should be read in one place, potentially given by SGD.
    m_frameMode = (ConfigValue)cfg("frameMode", "true");

    argvector<ConfigValue> inputs = cfg("input");
    if (inputs.size() != 1)
        LogicError("MLFDataDeserializer supports a single input stream only.");

    std::wstring precision = cfg(L"precision", L"float");;
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? ElementType::tfloat : ElementType::tdouble;

    ConfigParameters input = inputs.front();
    auto inputName = input.GetMemberIds().front();

    ConfigParameters streamConfig = input(inputName);
    ConfigHelper config(streamConfig);

    m_dimension = config.GetLabelDimension();

    m_withPhoneBoundaries = streamConfig(L"phoneBoundaries", false);
    if (m_frameMode && m_withPhoneBoundaries)
        LogicError("frameMode and phoneBoundaries are not supposed to be used together.");

    wstring labelMappingFile = streamConfig(L"labelMappingFile", L"");
    InitializeChunkDescriptions(corpus, config, labelMappingFile, m_dimension);
    InitializeStream(inputName, m_dimension);
}

MLFDataDeserializer::MLFDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& labelConfig, const wstring& name)
    : DataDeserializerBase(false)
{
    // The frame mode is currently specified once per configuration,
    // not in the configuration of a particular deserializer, but on a higher level in the configuration.
    // Because of that we are using find method below.
    m_frameMode = labelConfig.Find("frameMode", "true");

    ConfigHelper config(labelConfig);

    config.CheckLabelType();
    m_dimension = config.GetLabelDimension();

    if (m_dimension > numeric_limits<IndexType>::max())
    {
        RuntimeError("Label dimension (%" PRIu64 ") exceeds the maximum allowed "
            "value (%" PRIu64 ")\n", m_dimension, (size_t)numeric_limits<IndexType>::max());
    }

    std::wstring precision = labelConfig(L"precision", L"float");;
    m_elementType = AreEqualIgnoreCase(precision, L"float") ? ElementType::tfloat : ElementType::tdouble;

    m_withPhoneBoundaries = labelConfig(L"phoneBoundaries", "false");

    wstring labelMappingFile = labelConfig(L"labelMappingFile", L"");
    InitializeChunkDescriptions(corpus, config, labelMappingFile, m_dimension);
    InitializeStream(name, m_dimension);
}

// Currently we create a single chunk only.
void MLFDataDeserializer::InitializeChunkDescriptions(CorpusDescriptorPtr corpus, const ConfigHelper& config, const wstring& stateListPath, size_t dimension)
{
    // TODO: Similarly to the old reader, currently we assume all Mlfs will have same root name (key)
    // restrict MLF reader to these files--will make stuff much faster without having to use shortened input files
    vector<wstring> mlfPaths = config.GetMlfPaths();

    if (!stateListPath.empty())
    {
        m_stateTable = std::make_shared<StateTable>();
        m_stateTable->ReadStateList(stateListPath);
    }

    for (const auto& path : mlfPaths)
    {
        auto file = fopenOrDie(path, L"rbS");
        auto indexer = std::make_shared<MLFIndexer>(file, true);
        indexer->Build(corpus);
        m_indexers.push_back(make_pair(path, indexer));
    }




    MLFUtterance description;
    size_t numClasses = 0;
    size_t totalFrames = 0;

    // TODO resize m_keyToSequence with number of IDs from string registry
    for (const auto& l : labels)
    {
        auto key = l.first;
        if (!corpus->IsIncluded(key))
            continue;

        size_t id = corpus->KeyToId(key);
        description.m_key.m_sequence = id;

        const auto& utterance = l.second;
        description.m_sequenceStart = m_classIds.size();
        uint32_t numberOfFrames = 0;

        vector<size_t> sequencePhoneBoundaries(m_withPhoneBoundaries ? utterance.size() : 0); // Phone boundaries of given sequence
        foreach_index(i, utterance)
        {
            if (m_withPhoneBoundaries)
                sequencePhoneBoundaries[i] = utterance[i].firstframe;
            const auto& timespan = utterance[i];
            if ((i == 0 && timespan.firstframe != 0) ||
                (i > 0 && utterance[i - 1].firstframe + utterance[i - 1].numframes != timespan.firstframe))
            {
                RuntimeError("Labels are not in the consecutive order MLF in label set: %s", l.first.c_str());
            }

            if (timespan.classid >= dimension)
            {
                RuntimeError("Class id %d exceeds the model output dimension %d.", (int)timespan.classid, (int)dimension);
            }

            if (timespan.classid != static_cast<msra::dbn::CLASSIDTYPE>(timespan.classid))
            {
                RuntimeError("CLASSIDTYPE has too few bits");
            }

            if (SEQUENCELEN_MAX < timespan.firstframe + timespan.numframes)
            {
                RuntimeError("Maximum number of sample per sequence exceeded.");
            }

            numClasses = max(numClasses, (size_t)(1u + timespan.classid));

            for (size_t t = timespan.firstframe; t < timespan.firstframe + timespan.numframes; t++)
            {
                m_classIds.push_back(timespan.classid);
                numberOfFrames++;
            }
        }

        if (m_withPhoneBoundaries)
            m_phoneBoundaries.push_back(sequencePhoneBoundaries);

        description.m_numberOfSamples = numberOfFrames;
        m_utteranceIndex.push_back(totalFrames);
        totalFrames += numberOfFrames;

        if (m_keyToSequence.size() <= description.m_key.m_sequence)
        {
            m_keyToSequence.resize(description.m_key.m_sequence + 1, SIZE_MAX);
        }
        assert(m_keyToSequence[description.m_key.m_sequence] == SIZE_MAX);
        m_keyToSequence[description.m_key.m_sequence] = m_utteranceIndex.size() - 1;
        m_numberOfSequences++;
    }
    m_utteranceIndex.push_back(totalFrames);

    m_totalNumberOfFrames = totalFrames;

    fprintf(stderr, "MLFDataDeserializer::MLFDataDeserializer: %" PRIu64 " utterances with %" PRIu64 " frames in %" PRIu64 " classes\n",
            m_numberOfSequences,
            m_totalNumberOfFrames,
            numClasses);

    if (m_frameMode)
    {
        // Initializing array of labels.
        m_categories.reserve(dimension);
        m_categoryIndices.reserve(dimension);
        for (size_t i = 0; i < dimension; ++i)
        {
            auto category = make_shared<CategorySequenceData>();
            m_categoryIndices.push_back(static_cast<IndexType>(i));
            category->m_indices = &(m_categoryIndices[i]);
            category->m_nnzCounts.resize(1);
            category->m_nnzCounts[0] = 1;
            category->m_totalNnzCount = 1;
            category->m_numberOfSamples = 1;
            if (m_elementType == ElementType::tfloat)
            {
                category->m_data = &s_oneFloat;
            }
            else
            {
                assert(m_elementType == ElementType::tdouble);
                category->m_data = &s_oneDouble;
            }
            m_categories.push_back(category);
        }
    }
}

void MLFDataDeserializer::InitializeStream(const wstring& name, size_t dimension)
{
    // Initializing stream description - a single stream of MLF data.
    StreamDescriptionPtr stream = make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = name;
    stream->m_sampleLayout = make_shared<TensorShape>(dimension);
    stream->m_storageType = StorageType::sparse_csc;
    stream->m_elementType = m_elementType;
    m_streams.push_back(stream);
}

// Currently MLF has a single chunk.
// TODO: This will be changed when the deserializer properly supports chunking.
ChunkDescriptions MLFDataDeserializer::GetChunkDescriptions()
{
    auto cd = make_shared<ChunkDescription>();
    cd->m_id = 0;
    cd->m_numberOfSequences = m_frameMode ? m_totalNumberOfFrames : m_numberOfSequences;
    cd->m_numberOfSamples = m_totalNumberOfFrames;
    return ChunkDescriptions{cd};
}

// Gets sequences for a particular chunk.
void MLFDataDeserializer::GetSequencesForChunk(ChunkIdType, vector<SequenceDescription>& result)
{
    UNUSED(result);
    LogicError("Mlf deserializer does not support primary mode - it cannot control chunking.");
}

ChunkPtr MLFDataDeserializer::GetChunk(ChunkIdType chunkId)
{
    UNUSED(chunkId);
    assert(chunkId == 0);
    return make_shared<MLFChunk>(this);
};

// Sparse labels for an utterance.
template <class ElemType>
struct MLFSequenceData : SparseSequenceData
{
    vector<ElemType> m_values;
    unique_ptr<IndexType[]> m_indicesPtr;

    MLFSequenceData(size_t numberOfSamples) :
        m_values(numberOfSamples, 1),
        m_indicesPtr(new IndexType[numberOfSamples])
    {
        if (numberOfSamples > numeric_limits<IndexType>::max())
        {
            RuntimeError("Number of samples in an MLFSequence (%" PRIu64 ") "
                "exceeds the maximum allowed value (%" PRIu64 ")\n",
                numberOfSamples, (size_t)numeric_limits<IndexType>::max());
        }

        m_nnzCounts.resize(numberOfSamples, static_cast<IndexType>(1));
        m_numberOfSamples = (uint32_t) numberOfSamples;
        m_totalNnzCount = static_cast<IndexType>(numberOfSamples);
        m_indices = m_indicesPtr.get();
    }

    MLFSequenceData(size_t numberOfSamples, const vector<size_t>& phoneBoundaries) :
        MLFSequenceData(numberOfSamples)
    {
        for (auto boundary : phoneBoundaries)
            m_values[boundary] = PHONE_BOUNDARY;
    }

    const void* GetDataBuffer() override
    {
        return m_values.data();
    }
};

void MLFDataDeserializer::GetSequenceById(size_t sequenceId, vector<SequenceDataPtr>& result)
{
    if (m_frameMode)
    {
        size_t label = m_classIds[sequenceId];
        assert(label < m_categories.size());
        result.push_back(m_categories[label]);
    }
    else
    {
        // Packing labels for the utterance into sparse sequence.
        size_t startFrameIndex = m_utteranceIndex[sequenceId];
        size_t numberOfSamples = m_utteranceIndex[sequenceId + 1] - startFrameIndex;
        SparseSequenceDataPtr s;
        if (m_elementType == ElementType::tfloat)
        {
            if (m_withPhoneBoundaries)
                s = make_shared<MLFSequenceData<float>>(numberOfSamples, m_phoneBoundaries.at(sequenceId));
            else
                s = make_shared<MLFSequenceData<float>>(numberOfSamples);
        }
        else
        {
            assert(m_elementType == ElementType::tdouble);
            if (m_withPhoneBoundaries)
                s = make_shared<MLFSequenceData<double>>(numberOfSamples, m_phoneBoundaries.at(sequenceId));
            else
                s = make_shared<MLFSequenceData<double>>(numberOfSamples);
        }

        for (size_t i = 0; i < numberOfSamples; i++)
        {
            size_t frameIndex = startFrameIndex + i;
            size_t label = m_classIds[frameIndex];
            s->m_indices[i] = static_cast<IndexType>(label);
        }
        result.push_back(s);
    }
}

bool MLFDataDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result)
{

    auto sequenceId = key.m_sequence < m_keyToSequence.size() ? m_keyToSequence[key.m_sequence] : SIZE_MAX;

    if (sequenceId == SIZE_MAX)
    {
        return false;
    }

    result.m_chunkId = 0;
    result.m_key = key;

    if (m_frameMode)
    {
        size_t index = m_utteranceIndex[sequenceId] + key.m_sample;
        result.m_indexInChunk = index;
        result.m_numberOfSamples = 1;
    }
    else
    {
        assert(result.m_key.m_sample == 0);
        result.m_indexInChunk = sequenceId;
        result.m_numberOfSamples = (uint32_t) (m_utteranceIndex[sequenceId + 1] - m_utteranceIndex[sequenceId]);
    }
    return true;
}

}}}
