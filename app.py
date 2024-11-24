from flask import Flask, request, render_template
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the Hugging Face question-answering model
model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# Context containing information about Canadian province capitals
context = """
Canada has 10 provinces and 3 territories. Here are their capitals:
- Alberta: Edmonton
- British Columbia: Victoria
- Manitoba: Winnipeg
- New Brunswick: Fredericton
- Newfoundland and Labrador: St. John's
- Nova Scotia: Halifax
- Ontario: Toronto
- Prince Edward Island: Charlottetown
- Quebec: Quebec City
- Saskatchewan: Regina
- Northwest Territories: Yellowknife
- Nunavut: Iqaluit
- Yukon: Whitehorse
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the user's question from the form
        question = request.form.get('text_input')

        if question:
            try:
                # Use the Hugging Face model to answer the question
                QA_input = {'question': question, 'context': context}
                result = nlp(QA_input)
                answer = result['answer']
                return render_template('index.html', question=question, answer=answer)
            except Exception as e:
                # Handle errors
                error = f"Error: {str(e)}"
                return render_template('index.html', error=error)
        else:
            # Handle empty input
            error = "Please enter a question."
            return render_template('index.html', error=error)

    # For GET request
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
