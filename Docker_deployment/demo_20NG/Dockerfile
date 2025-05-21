# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file (we'll create this inline) and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data
RUN python -m nltk.downloader stopwords

# Copy the application files, including the .joblib files
COPY app.py .
COPY tf_idf_VotingClassifier.joblib .
COPY tfidf_vectorizer.joblib .

# Expose the port that Gradio will use
EXPOSE 7860

# Set environment variables for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Command to run the Gradio app
CMD ["python", "app.py"]