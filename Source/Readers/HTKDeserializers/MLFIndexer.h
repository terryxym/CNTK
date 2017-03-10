//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/noncopyable.hpp>

#include "Indexer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class MLFIndexer : private boost::noncopyable
    {
    public:
        MLFIndexer(FILE* file, bool isPrimary, size_t chunkSize = 32 * 1024 * 1024, size_t bufferSize = 2 * 1024 * 1024);

        void Build(CorpusDescriptorPtr corpus);

        // Returns input data index (chunk and sequence metadata)
        const Index& GetIndex() const { return m_index; }

    private:
        FILE* m_file;

        int64_t m_fileOffsetStart;
        int64_t m_fileOffsetEnd;

        std::unique_ptr<char[]> m_buffer;
        const size_t m_bufferSize;
        const char* m_bufferStart;
        const char* m_bufferEnd;
        const char* m_pos; // buffer index

        bool m_done; // true, when all input was processed
        Index m_index;

        // fills up the buffer with data from file, all previously buffered data
        // will be overwritten.
        void RefillBuffer();

        // Moves the buffer position to the beginning of the next line.
        // Returns true if the line contains a single dot.
        bool SkipLine();

        // Returns current offset in the input file (in bytes).
        int64_t GetFileOffset() const { return m_fileOffsetStart + (m_pos - m_bufferStart); }


        bool TryGetSequenceId(size_t& id, std::function<size_t(const std::string&)> keyToId);

        std::string m_lastElementsInPreviousBuffer;

        void GetMLF();
    };

}}} // namespace
