import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
    df.head()
    return (df,)


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df.sample()['clean_comment'].values
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell
def _(df):
    df[df['clean_comment'].isna()]
    return


@app.cell
def _(df):
    df[df['clean_comment'].isna()]['category'].value_counts()
    return


@app.cell
def _(df):
    df.dropna(inplace=True)
    return


@app.cell
def _(df):
    df.duplicated().sum()
    return


@app.cell
def _(df):
    df[df.duplicated()]
    return


@app.cell
def _(df):
    df.drop_duplicates(inplace=True)
    return


@app.cell
def _(df):
    df.duplicated().sum()
    return


@app.cell
def _(df):
    df[(df["clean_comment"].str.strip() == '')]
    return


@app.cell
def _(df):
    df_1 = df[~(df['clean_comment'].str.strip() == '')]
    return (df_1,)


@app.cell
def _(df_1):
    df_1['clean_comment'] = df_1['clean_comment'].str.lower()
    return


@app.cell
def _(df_1):
    df_1.head()
    return


@app.cell
def _(df_1):
    df_1[df_1['clean_comment'].apply(lambda x: x.endswith(' ') or x.startswith(' '))]
    return


@app.cell
def _(df_1):
    df_1['clean_comment'] = df_1['clean_comment'].str.strip()
    df_1['clean_comment'].apply(lambda x: x.endswith(' ') or x.startswith(' ')).sum()
    return


@app.cell
def _(df_1):
    url_pattern = 'http[s]?//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    comments_with_urls = df_1[df_1['clean_comment'].str.contains(url_pattern, regex=True)]
    comments_with_urls.head()
    return


@app.cell
def _(df_1):
    # comments with new lines
    comments_with_newlines = df_1[df_1['clean_comment'].str.contains('\n')]
    comments_with_newlines.head()
    return


@app.cell
def _(df_1):
    df_1['clean_comment'] = df_1['clean_comment'].str.replace('\n', ' ', regex=True)
    comments_with_newline_remaining = df_1[df_1['clean_comment'].str.contains('\n')]
    comments_with_newline_remaining
    return


@app.cell
def _(df_1):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.countplot(data=df_1, x='category')
    return plt, sns


@app.cell
def _(df_1):
    df_1['category'].value_counts(normalize=True).mul(100).round(2)
    return


@app.cell
def _(df_1):
    df_1['word_count'] = df_1['clean_comment'].apply(lambda x: len(x.split()))
    return


@app.cell
def _(df_1):
    df_1.sample(10)
    return


@app.cell
def _(df_1):
    df_1['word_count'].describe()
    return


@app.cell
def _(df_1, sns):
    sns.displot(df_1['word_count'], kde=True)
    return


@app.cell
def _(df_1, plt, sns):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_1[df_1['category'] == 1]['word_count'], label='Positive', fill=True)
    sns.kdeplot(df_1[df_1['category'] == 0]['word_count'], label='Negative', fill=True)
    sns.kdeplot(df_1[df_1['category'] == -1]['word_count'], label='Neutral', fill=True)
    plt.title('Word Count Distribution by Sentiment Category')
    plt.xlabel('Word Count')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    return


@app.cell
def _(df_1, sns):
    sns.boxplot(df_1['word_count'])
    return


@app.cell
def _(df_1, plt, sns):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='category', y='word_count', data=df_1)
    plt.title('Boxplot of Word Count by Sentiment Category')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Word Count')
    plt.xticks(ticks=[-1, 0, 1], labels=['Negative', 'Neutral', 'Positive'])
    plt.show()
    return


@app.cell
def _(df_1, plt, sns):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='category', y='word_count', alpha=0.5, data=df_1)
    plt.title('Scatter Plot of Word Count vs Sentiment Category')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Word Count')
    plt.xticks(ticks=[-1, 0, 1], labels=['Negative', 'Neutral', 'Positive'])
    plt.show()
    return


@app.cell
def _(df_1, sns):
    sns.barplot(x='category', y='word_count', data=df_1)
    return


@app.cell
def _(df_1):
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    df_1['num_stop_words'] = df_1['clean_comment'].apply(lambda x: len([word for word in x.split() if word in stop_words]))
    return nltk, stop_words, stopwords


@app.cell
def _(df_1):
    df_1.sample(6)
    return


@app.cell
def _(df_1, plt, sns):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_1['num_stop_words'], kde=True)
    plt.title('Distribution of Number of Stop Words in Comments')
    plt.xlabel('Number of Stop Words')
    plt.ylabel('Frequency')
    plt.show()
    return


@app.cell
def _(df_1, plt, sns):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_1[df_1['category'] == 1]['num_stop_words'], label='Positive', fill=True)
    sns.kdeplot(df_1[df_1['category'] == 0]['num_stop_words'], label='Negative', fill=True)
    sns.kdeplot(df_1[df_1['category'] == -1]['num_stop_words'], label='Neutral', fill=True)
    plt.title('Distribution of Number of Stop Words by Sentiment Category')
    plt.xlabel('Number of Stop Words')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    return


@app.cell
def _(df_1, pd, plt, sns, stop_words):
    from collections import Counter
    all_stop_words = [word for comment in df_1['clean_comment'] for word in comment.split() if word in stop_words]
    most_common_stop_words = Counter(all_stop_words).most_common(25)
    top_25_df = pd.DataFrame(most_common_stop_words, columns=['Stop Word', 'Frequency'])
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Frequency', y='Stop Word', data=top_25_df, hue='Stop Word', palette='viridis')
    plt.title('Top 25 Most Common Stop Words in Comments')
    plt.xlabel('Frequency')
    plt.ylabel('Stop Word')
    plt.show()
    return (Counter,)


@app.cell
def _(df_1):
    df_1['num_chars'] = df_1['clean_comment'].apply(len)
    df_1.head()
    return


@app.cell
def _(df_1):
    df_1['num_chars'].describe()
    return


@app.cell
def _(Counter, df_1, pd):
    _all_text = ' '.join(df_1['clean_comment'])
    char_freq = Counter(_all_text)
    char_freq_df = pd.DataFrame(char_freq.items(), columns=['Character', 'Frequency']).sort_values(by='Frequency', ascending=False)
    return (char_freq_df,)


@app.cell
def _(char_freq_df):
    char_freq_df['Character'].values
    return


@app.cell
def _(df_1):
    df_1['num_punctuation_chars'] = df_1['clean_comment'].apply(lambda x: sum([1 for char in x if char in '.,!?;:()[]{}"\'-']))
    df_1.sample(5)
    return


@app.cell
def _(df_1):
    df_1['num_punctuation_chars'].describe()
    return


@app.cell
def _(df_1, pd, plt, sns):
    from sklearn.feature_extraction.text import CountVectorizer

    def get_top_ngrams(corpus, ngram_size=2, top_k=25):
        vec = CountVectorizer(ngram_range=(ngram_size, ngram_size), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    top_25_bigrams = get_top_ngrams(df_1['clean_comment'], ngram_size=2, top_k=25)
    top_25_bigrams_df = pd.DataFrame(top_25_bigrams, columns=['bigram', 'Frequency'])
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Frequency', y='bigram', data=top_25_bigrams_df, hue='bigram', palette='viridis')
    plt.title('Top 25 Most Common Bigrams in Comments')
    plt.xlabel('Frequency')
    plt.ylabel('Bigram')
    plt.show()
    return (CountVectorizer,)


@app.cell
def _(CountVectorizer, df_1, pd, plt, sns):
    # Create a function to extract the top 25 trigrams
    def get_top_trigrams(corpus, n=None):
        vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]
    top_25_trigrams = get_top_trigrams(df_1['clean_comment'], 25)
    # Get the top 25 trigrams
    top_25_trigrams_df = pd.DataFrame(top_25_trigrams, columns=['trigram', 'count'])
    plt.figure(figsize=(12, 8))
    # Convert the trigrams into a DataFrame for plotting
    sns.barplot(data=top_25_trigrams_df, x='count', y='trigram', palette='coolwarm')
    plt.title('Top 25 Most Common Trigrams')
    # Plot the countplot for the top 25 trigrams
    plt.xlabel('Count')
    plt.ylabel('Trigram')
    plt.show()
    return


@app.cell
def _(df_1):
    import re
    df_1['clean_comment'] = df_1['clean_comment'].apply(lambda x: re.sub('[^A-Za-z0-9\\s!?.,]', '', str(x)))
    return


@app.cell
def _(Counter, df_1, pd):
    _all_text = ' '.join(df_1['clean_comment'])
    char_frequency = Counter(_all_text)
    char_frequency_df = pd.DataFrame(char_frequency.items(), columns=['character', 'frequency']).sort_values(by='frequency', ascending=False)
    char_frequency_df
    return


@app.cell
def _(df_1):
    df_1.head()
    return


@app.cell
def _(df_1, stopwords):
    stop_words_1 = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    # Defining stop words but keeping essential ones for sentiment analysis
    # Remove stop words from 'clean_comment' column, retaining essential ones
    df_1['clean_comment'] = df_1['clean_comment'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words_1]))
    return


@app.cell
def _(df_1, nltk):
    from nltk.stem import WordNetLemmatizer
    nltk.download('wordnet')
    # lemmatization => is the way of getting root word like play plays playing play is the root word
    lemmatizer = WordNetLemmatizer()
    df_1['clean_comment'] = df_1['clean_comment'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    # Define the lemmatizer
    # Apply lemmatization to the 'clean_comment_no_stopwords' column
    df_1.head()
    return


@app.cell
def _(df_1, plt):
    from wordcloud import WordCloud

    def _plot_word_cloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    _plot_word_cloud(df_1['clean_comment'])
    return (WordCloud,)


@app.cell
def _(WordCloud, df_1, plt):
    def _plot_word_cloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    _plot_word_cloud(df_1[df_1['category'] == 1]['clean_comment'])
    return


@app.cell
def _(WordCloud, df_1, plt):
    def _plot_word_cloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    _plot_word_cloud(df_1[df_1['category'] == 0]['clean_comment'])
    return


@app.cell
def _(WordCloud, df_1, plt):
    def _plot_word_cloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    _plot_word_cloud(df_1[df_1['category'] == -1]['clean_comment'])
    return


@app.cell
def _(Counter, df_1, plt, sns):
    def plot_top_n_words(df, n=20):
        """Plot the top N most frequent words in the dataset."""
        words = ' '.join(df['clean_comment']).split()  # Flatten all words in the content column
        counter = Counter(words)
        most_common_words = counter.most_common(n)
        words, counts = zip(*most_common_words)  # Get the top N most common words
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words))
        plt.title(f'Top {n} Most Frequent Words')
        plt.xlabel('Frequency')  # Split the words and their counts for plotting
        plt.ylabel('Words')
        plt.show()
    # Example usage
    plot_top_n_words(df_1, n=50)  # Plot the top N words
    return


@app.cell
def _(df_1, plt):
    def plot_top_n_words_by_category(df, n=20, start=0):
        """Plot the top N most frequent words in the dataset with stacked hue based on sentiment category."""
        word_category_counts = {}  # Flatten all words in the content column and count their occurrences by category
        for idx, row in df.iterrows():
            words = row['clean_comment'].split()
            category = row['category']
            for word in words:
                if word not in word_category_counts:  # Assuming 'category' column exists for -1, 0, 1 labels
                    word_category_counts[word] = {-1: 0, 0: 0, 1: 0}
                word_category_counts[word][category] = word_category_counts[word][category] + 1
        total_word_counts = {word: sum(counts.values()) for word, counts in word_category_counts.items()}
        most_common_words = sorted(total_word_counts.items(), key=lambda x: x[1], reverse=True)[start:start + n]  # Initialize counts for each sentiment category
        top_words = [word for word, _ in most_common_words]
        word_labels = top_words  # Increment the count for the corresponding sentiment category
        negative_counts = [word_category_counts[word][-1] for word in top_words]
        neutral_counts = [word_category_counts[word][0] for word in top_words]
        positive_counts = [word_category_counts[word][1] for word in top_words]  # Get total counts across all categories for each word
        plt.figure(figsize=(12, 8))
        bar_width = 0.75
        plt.barh(word_labels, negative_counts, color='red', label='Negative (-1)', height=bar_width)  # Get the top N most frequent words across all categories
        plt.barh(word_labels, neutral_counts, left=negative_counts, color='gray', label='Neutral (0)', height=bar_width)
        plt.barh(word_labels, positive_counts, left=[i + j for i, j in zip(negative_counts, neutral_counts)], color='green', label='Positive (1)', height=bar_width)
        plt.xlabel('Frequency')
        plt.ylabel('Words')  # Prepare data for plotting
        plt.title(f'Top {n} Most Frequent Words with Stacked Sentiment Categories')
        plt.legend(title='Sentiment', loc='lower right')
        plt.gca().invert_yaxis()
        plt.show()
    plot_top_n_words_by_category(df_1, n=20)  # Plot the stacked bar chart  # Plot negative, neutral, and positive counts in a stacked manner  # Invert y-axis to show the highest frequency at the top
    return


if __name__ == "__main__":
    app.run()
