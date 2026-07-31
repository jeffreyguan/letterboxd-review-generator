# Letterboxd Review Generator
A GPT-style model that generates Letterboxd-style reviews. Trained on two datasets: ~90k reviews of the Letterboxd Top 250 (https://www.kaggle.com/datasets/riyosha/letterboxd-movie-reviews-90000), and ~87k reviews of the Letterboxd Worst 250 (https://www.kaggle.com/datasets/riyosha/letterboxd-worst-250-movie-reviews-87k), totalling to ~177k reviews. 

First, I trained a subword bpe tokenizer on all the reviews with a vocab size of 8000. There were also 13 special tokens: 10 for ratings (0.5-->5 stars), two delimiter tokens for titles, and an end of text token. The delimiter token is used to "surround" (there has to be a better word for this) the title, hopefully allowing the model to associate the title of a movie if its mentioned in the review (eg, if a review of the dark knight starts with "the dark knight is tremendous...", i dont want the model to start randomly namedropping the dark knight, i want it to namedrop the title of whatever movie was given). 

These special tokens also allow someone to specify a given rating and title before generating. When a rating and title is not specified, end of text is used as the first token. 

The model itself is largely based on similar "from scratch" GPT models such as Karpathy's nanoGPT. I wanted more of a "from-scratch" feel, so I didn't use PyTorch's built in Transformer class. The model is a decoder only transformer. 

## Limitations
Currently, the model that I have trained speaks "half-gibberish". The sentences are mostly gramatically correct, it knows when to use certain words, but it doesnt make a coherent review. For instance, it will call a movie a masterpiece in one sentence and then terrible in the next. 

I think there are a couple reasons for this. The first is that the model size is very small right now. I chose small parameter numbers because I'm testing and also because I'm training this model on my M4 Mac Air. When I get home (and access to my RTX 5070), I'll train a larger model. 

My dataset is also probably too small to train an actually good model. The Letterboxd API is currently restricted (it states you can't use it for ML purposes or for personal projects), so I would have to scrape data myself for a more comprehensive data, which would take a lot of time. 

Besides the size of my dataset, the dataset is also very skewed. Because the dataset is from the Top and Bottom 250 movies, most of the ratings are either really high or really now. As a result, the model is probably worse at knowing that a "mid" review is. A better dataset would have a more even distribution of ratings. 

Below I asked claude to write a guide. 

## Guide

### Setup
```
pip install -r requirements.txt
```

### 1. Data
Drop your review CSVs into `data/`. Each CSV needs a title column, a rating column, and a review-text column — `process_data.py` sniffs common header names automatically (e.g. `title`/`movie`/`film`, `rating`/`stars`/`score`, `review`/`text`/`content`), and it handles both numeric ratings and Letterboxd's star-glyph format (`★★★½`). If your columns use unusual names, edit the `*_CANDIDATES` lists at the top of `process_data.py`.

### 2. Process data + train the tokenizer
```
python src/process_data.py
```
This cleans/combines the CSVs and trains a BPE tokenizer on the review text, saving `data/tokenizer.model` and `data/tokenizer.vocab`. Pass `--vocab-size` to change the vocab (default 2000 merges; the project's trained model uses 8000). If you change the vocab size, update `vocab_size` in `train.py`'s `GPTConfig` (merges + 13 special tokens) to match, or the model's embedding/output layers won't line up with the tokenizer.

### 3. Train
```
python src/train.py
```
Trains for a fixed number of epochs (`epochs` near the bottom of `train.py`), saving the best-so-far model to `checkpoints/ckpt_best.pth` whenever test loss improves, with early stopping after 3 non-improving epochs. Rerunning `train.py` automatically resumes from that checkpoint (model + optimizer state, epoch count, best loss) rather than starting over — handy for picking training back up across multiple sessions.

For a long run you don't want to babysit, background it and keep the machine from sleeping (macOS):
```
cd src
caffeinate -s nohup python -u train.py >> train.log 2>&1 &
disown
```
Then `tail -f train.log` to watch progress. Running inside `tmux`/`screen` first also protects the job if your terminal disconnects.

### 4. Generate
```
python src/generate.py --title "The Dark Knight" --rating 5.0
```
- Omit `--title`/`--rating` for unconditional generation (starts from the end-of-text token).
- `--temperature` controls randomness (lower = safer/more repetitive, higher = more chaotic).
- `--top-k` restricts sampling to the k most likely tokens at each step (default 40).
- `--checkpoint` points at a different checkpoint file if you want to compare models.
