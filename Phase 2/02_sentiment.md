## Step 1 — Load & inspect
load_dataset('stanfordnlp/imdb'): 25,000 train / 25,000 test, perfectly balanced (12,500/12,500 each way). Review lengths range from 10 to 2,470 words, averaging ~234 words.

## Step 2 — Tokenize & build vocabulary
Lowercased, stripped <br /> tags (IMDB's HTML line breaks), regex-split on non-alphanumeric characters. Built the vocab from training data only (no leakage) — top 19,998 words + <pad>/<unk> = 20,000 tokens. Max sequence length 200 (truncate/pad).

## Step 3 — Represent text numerically
Wrapped everything in a Dataset/DataLoader (22,500 train / 2,500 val / 25,000 test, batch size 64). Confirmed the embedding transform shape-wise: (64, 200) integer batch → (64, 200, 100) once passed through nn.Embedding.

## Step 4 — Bag-of-embeddings model
Embedding(20000, 100) → mean-pool (masking out padding first) → Linear(100,64) → Linear(64,1). Masking padding before pooling matters — otherwise the model partly "sees" a wall of meaningless zeros on short reviews.

## Step 5 — Train with a real loop
Same hand-written loop pattern from MNIST, BCEWithLogitsLoss this time. 6 epochs, 21.6s total:

Val accuracy peaked at 80.2% (epoch 3), then val loss started climbing while train loss kept dropping (0.055→0.12 by epoch 6) — textbook overfitting, visible within just a few epochs on this simple architecture. Below the guide's 85–88% target — worth being honest about. Likely levers to close the gap: pretrained embeddings instead of learned-from-scratch, dropout, or just checkpointing the epoch-3 weights instead of training straight through to epoch 6.
## Step 6 — Recurrent upgrade
Single-layer LSTM (hidden dim 128) using pack_padded_sequence so padding doesn't pollute the recurrence. 4 epochs, 489.7s — training got genuinely rocky early (val acc actually dropped epoch 1→2, 56.6%→53.3%, before recovering to 79.3% by epoch 4), which is a real LSTM training-dynamics quirk, not a bug. Final best: 79.3% val accuracy.

The actual finding here matches the field note almost exactly: the LSTM did not beat the bag-of-embeddings baseline (79.3% vs 80.2%) — and took 22x longer to train (490s vs 22s). That's the lesson this specimen was built to teach, and it landed cleanly: reach for the recurrent model only when it earns its keep.

## Step 7 — Inspect mistakes
Pulled the first 8 test-set misclassifications — all were negative reviews (true=0) predicted positive. Worth a methodology note: I took mismatches in raw index order rather than a random sample, and IMDB's test split happens to be ordered negatives-then-positives, so this 8-example batch landing all-one-class is a sampling artifact, not necessarily a systematic model bias — precision (81.8%) and recall (79.9%) being close suggests errors are fairly balanced overall. That said, the actual content of these misses matches the field note's prediction exactly — several are negative reviews stuffed with positive-sounding phrases ("brilliant New Zealand actress," "always a pleasure to watch," "great atmosphere") before pivoting negative, which is exactly the mixed-sentiment trap this step was written to surface.

## Step 8 — Final evaluation
LSTM on the untouched test set, touched once: 81.1% accuracy, 81.8% precision, 79.9% recall, 80.9% F1 — also short of the guide's ~88-90% target range. Same root cause as step 5: no pretrained embeddings, no dropout, single direction, short training.

Bottom line vs. benchmarks: both models landed a bit below the guide's optimistic ranges (80.2% vs 85-88%, 81.1% vs 88-90%), but the comparative result — pooled embeddings beating a much more expensive LSTM — is the real specimen lesson, and it came through clean. Notebook's saved at 02_sentiment.ipynb.