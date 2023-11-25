# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources (if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
# Assuming you have a CSV file named 'cleantech_dataset.csv'
df = pd.read_csv('cleantech_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Data Collection and Cleaning
# Task 1: Acquire the Dataset
# - Download the dataset from the Kaggle link you provided.
# - Load the dataset into a Pandas DataFrame or any other suitable data structure.

# Task 2: Data Cleaning
# - Remove irrelevant information: Identify and drop any columns that may not contribute to your analysis.
# - Handle missing values if any.
# - Remove duplicates.
# - Handle special characters or formatting issues.
df.set_index('id', inplace=True)

# Text Preprocessing
# Task 3: Tokenization
# - Tokenize the text data into words or phrases. Use a tokenizer library or built-in functions.

# Task 4: Stemming or Lemmatization
# - Apply stemming or lemmatization to reduce words to their root form. This helps in standardizing the text.

# Task 5: Remove Stop Words and Non-Informative Terms
# - Remove common stop words and terms that may not contribute much to the analysis.
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stop words and non-alphabetic words
    words = [ps.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    
    return ' '.join(words)

# Apply text preprocessing to the 'content' column
df['processed_content'] = df['content'].apply(preprocess_text)

# Exploratory Data Analysis (EDA)
# Task 6: Basic Statistics
# - Compute basic statistics such as word counts, document lengths, and term frequencies.
word_counts = df['processed_content'].apply(lambda x: len(x.split()))
document_lengths = df['processed_content'].apply(lambda x: len(x))

# Task 7: Word Clouds
# - Create word clouds to visualize the most frequent terms.
wordcloud = WordCloud(width=800, height=400, max_words=150, background_color='white').generate(' '.join(df['processed_content']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Task 8: Histograms/Bar Charts
# - Plot histograms or bar charts to analyze the distribution of major cleantech topics and categories.
plt.figure(figsize=(10, 5))
plt.hist(word_counts, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# Task 9: Scatter Plot for Document Lengths
# - Visualize the distribution of texts using scatter plots or other suitable visualization techniques.
plt.figure(figsize=(10, 5))
plt.scatter(document_lengths, word_counts, color='coral', alpha=0.5)
plt.title('Scatter Plot of Document Lengths vs. Word Counts')
plt.xlabel('Document Length')
plt.ylabel('Word Count')
plt.show()
