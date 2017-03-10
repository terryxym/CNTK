//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#define _CRT_SECURE_NO_WARNINGS
#include <inttypes.h>
#include "MLFIndexer.h"

using std::string;

const static char ROW_DELIMITER = '\n';

namespace Microsoft { namespace MSR { namespace CNTK {

    MLFIndexer::MLFIndexer(FILE* file, bool primary, size_t chunkSize, size_t bufferSize) :
        m_bufferSize(bufferSize),
        m_file(file),
        m_fileOffsetStart(0),
        m_fileOffsetEnd(0),
        m_buffer(new char[bufferSize + 1]),
        m_bufferStart(nullptr),
        m_bufferEnd(nullptr),
        m_pos(nullptr),
        m_done(false),
        m_index(chunkSize, primary)
    {
        if (!m_file)
            RuntimeError("Input file not open for reading");
    }

    void MLFIndexer::RefillBuffer()
    {
        if (!m_done)
        {
            if (m_bufferEnd - m_bufferStart > 3)
                m_lastElementsInPreviousBuffer.assign(m_bufferEnd - 3, m_bufferEnd); // remember last three elements.
            else
                m_lastElementsInPreviousBuffer.append(m_bufferStart, m_bufferEnd); // otherwise add 1 or 2 elements.

            size_t bytesRead = fread(m_buffer.get(), 1, m_bufferSize, m_file);
            if (bytesRead == (size_t)-1)
                RuntimeError("Could not read from the input file.");

            if (bytesRead == 0)
                m_done = true;
            else
            {
                m_fileOffsetStart = m_fileOffsetEnd;
                m_fileOffsetEnd += bytesRead;
                m_bufferStart = m_buffer.get();
                m_pos = m_bufferStart;
                m_bufferEnd = m_bufferStart + bytesRead;
            }
        }
    }

    void MLFIndexer::Build(CorpusDescriptorPtr corpus)
    {
        if (!m_index.IsEmpty())
            return;

        m_index.Reserve(filesize(m_file));

        RefillBuffer(); // read the first block of data
        if (m_done)
            RuntimeError("Input file is empty");

        if ((m_bufferEnd - m_bufferStart > 3) &&
            (m_bufferStart[0] == '\xEF' && m_bufferStart[1] == '\xBB' && m_bufferStart[2] == '\xBF'))
        {
            // input file contains UTF-8 BOM value, skip it.
            m_pos += 3;
            m_fileOffsetStart += 3;
            m_bufferStart += 3;
        }

        GetMLF();

        size_t id = 0;
        int64_t offset = GetFileOffset();

        // read the very first sequence id
        if (!TryGetSequenceId(id, corpus->KeyToId))
            RuntimeError("Expected a sequence id at the offset %" PRIi64 ", none was found.", offset);

        SequenceDescriptor sd = {};
        sd.m_fileOffsetBytes = offset;

        bool previousDot = false;
        size_t previousId = id;
        while (!m_done)
        {
            previousDot = SkipLine(); // ignore whatever is left on this line.
            offset = GetFileOffset(); // a new line starts at this offset;
            sd.m_numberOfSamples++;

            if (!m_done && previousDot && TryGetSequenceId(id, corpus->KeyToId))
            {
                // found a new sequence, which starts at the [offset] bytes into the file
                sd.m_byteSize = offset - sd.m_fileOffsetBytes;
                sd.m_key.m_sequence = previousId;
                m_index.AddSequence(sd);

                sd = {};
                sd.m_fileOffsetBytes = offset;
                previousId = id;
            }
        }

        // calculate the byte size for the last sequence
        sd.m_byteSize = m_fileOffsetEnd - sd.m_fileOffsetBytes;
        sd.m_key.m_sequence = previousId;
        m_index.AddSequence(sd);
    }

    bool MLFIndexer::SkipLine()
    {
        // Need to check if the current line is a \n.[\r]\n
        bool dot = false;
        while (!m_done)
        {
            m_pos = (char*)memchr(m_pos, ROW_DELIMITER, m_bufferEnd - m_pos);

            if (m_pos)
            {
                if (m_pos - m_bufferStart >= 3)
                {
                    dot = ((*(m_pos - 1) == '.') && (*(m_pos - 2) == ROW_DELIMITER)) ||
                        ((*(m_pos - 1) == '\r') && (*(m_pos - 2) == '.') && (*(m_pos - 3) == ROW_DELIMITER));
                }
                else
                {
                    // Check whether the pattern is on the border of the buffer.
                    auto line = m_lastElementsInPreviousBuffer;
                    line.append(m_bufferStart, m_pos);
                    auto pos = m_lastElementsInPreviousBuffer.end();
                    if (line.size() >= 2)
                        dot |= ((*(pos - 1) == '.') && (*(pos - 2) == ROW_DELIMITER));
                    if (line.size() >= 2)
                        dot |= ((*(pos - 1) == '\r') && (*(pos - 2) == '.') && (*(pos - 3) == ROW_DELIMITER));
                }

                //found a new-line character
                if (++m_pos == m_bufferEnd)
                    RefillBuffer();

                return dot;
            }

            RefillBuffer();
        }
    }

    void MLFIndexer::GetMLF()
    {
        auto counter = 0;
        std::string mlf;
        while (!m_done && counter < 7)
        {
            mlf += *m_pos++;
            m_fileOffsetStart++;
            if (m_pos == m_bufferEnd)
                RefillBuffer();
        }

        if (mlf != "#!MLF!#")
            RuntimeError("Expected MLF header was not found.");
    }

    bool MLFIndexer::TryGetSequenceId(size_t& id, std::function<size_t(const std::string&)> keyToId)
    {
        bool found = false;
        id = 0;
        std::string key;
        key.reserve(256);
        while (!m_done)
        {
            char c = *m_pos;
            if (!isdigit(c) && !isalpha(c))
            {
                if (found)
                    id = keyToId(key);
                return found;
            }

            key += c;
            found = true;
            ++m_pos;

            if (m_pos == m_bufferEnd)
                RefillBuffer();
        }

        // reached EOF without hitting the pipe character,
        // ignore it for not, parser will have to deal with it.
        return false;
    }

}}}
