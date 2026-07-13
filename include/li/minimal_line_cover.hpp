/*
* minimal_line_cover.hpp - The EXACT epsilon coverability solver
*
*
* Does an OptimalPLR scan that answers "are theys keys coverable by a single line within eps?"
* and if not, "where are the forced cuts and what line covers each piece?"
* Wrapings streamingCone
* Remember rank locality
*
* Only called on a break
* Remember due to irreducibility lemma, a single segment splits into 1 <= n <= 3 segments
*/

#pragma once

#include "types.hpp"
#include "model.hpp"
#include "segmentation.hpp"
#include "status.hpp"

#include <vector>
#include <span>
#include <cstddef>

namespace li::detail {

enum class LineCoverStatus { COVERABLE, SPLIT };

struct LineCoverPiece {
    std::size_t begin;
    std::size_t end;
    LinearModel model;
};

struct LineCoverResult {
    LineCoverStatus status;
    LinearModel model;
    std::vector<LineCoverPiece> pieces;
};

// TODO(perf):
// if we really need some performance, pieces is a heap allocated vector that gets pushed to
// a bunch here.
// can use a smallvec and allocate 4 pieces. SmallVector<LineCoverPiece, 4>
// but this is constant, so measure before and after if i wnt to do tis
inline LineCoverResult minimal_line_cover(std::span<const Key> sorted_keys, double epsilon) {
    LI_ASSERT(!sorted_keys.empty());

    LineCoverResult result;
    std::size_t piece_start = 0;
    StreamingCone cone(sorted_keys[0], epsilon);

    for (std::size_t i = 0; i < sorted_keys.size(); ++i) {
        std::size_t local_rank = i - piece_start;

        if (!cone.try_extend(sorted_keys[i], local_rank)) {
            result.pieces.push_back(LineCoverPiece{ piece_start, i, cone.finalize().model });

            piece_start = i;
            cone = StreamingCone(sorted_keys[i], epsilon);
            cone.try_extend(sorted_keys[i], 0);
        }
    }

    result.pieces.push_back(LineCoverPiece{ piece_start, sorted_keys.size(), cone.finalize().model });

    if (result.pieces.size() == 1) {
        result.status = LineCoverStatus::COVERABLE;
        result.model = result.pieces.front().model;
    } else {
        result.status = LineCoverStatus::SPLIT;
    }

    return result;
}

}