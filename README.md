# CEDAR

### **C**oncurrent, **E**psilon-bounded, **D**ynamic, **A**rray-based index, with in-place model **R**epair

CEDAR is an in memory learned index that maintains a strict error bound as a hard invariant: every key is within \epsilon of its model's prediction at all times, with no fallback Updates run in place: inserts, real (not tombstone) deletes, and merges complete inline with per op worst case bounds. When model drift from inserts/deletes break a segment's epsilon band, the model is repaired locally using refit, split, or merge, rather than a deferred process in the background. Segments store keys in flat sorted arrays under a small B+tree of models.

Against a tuned in memory B+tree (tlx), CEDAR currently serves point lookups ~1.8x faster at p50, matches it on sorted inserts and range scanes, and uses almost 10x less memory overhead. Under churn with real deletes and inserts, range scan p99 is ~3x better because merging keeps survivor runs dense where B+tree leaves fragment, and the worst ase delete is ~100microseconds vs. B+tree's ~4milliseconds, because no operation ever defers work. The price of this is mutations, shuffled inserts and deletes run ~1.2-1.6x slower at  p50, the cost of keeping every key with epsilon.

Concurrency and disk management is in development.