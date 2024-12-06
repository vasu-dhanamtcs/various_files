import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
#from dotenv import load_dotenv
#load_dotenv()

# print(os.getenv('dataset_path'))
# Step 1: Load the dataset
data = pd.read_csv("D:/Py/Django/chatbot_python3.13.0/chat_bot_application/chatbot_data.csv")

# Step 2: Split the dataset into training and test sets
X = data['text']  # Input column (user queries)
y = data['intent']  # Target column (intents)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Convert text into numbers using CountVectorizer
vectorizer = CountVectorizer(lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)  # Fit on training data
X_test_vec = vectorizer.transform(X_test)       # Transform test data


model = MultinomialNB()

# Step 2: Train the model
model.fit(X_train_vec, y_train)

# Step 3: Test the model
accuracy = model.score(X_test_vec, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")


responses = {
    "greeting": "Hello! How can I assist you today?",
    "stock_audit_query": "We have implemented an enhanced Stock Audit Report form in the back office, located under the Reports tab. This update includes a new PDF button that allows you to download detailed stock audit reports for better inventory management and compliance.",
    "generate_stock_audit": 'Follow these steps to generate the report:\r\n\n1.Navigate to the Reports Tab:\n\n-Log in to the back office system.\n\n-Click on the Reports tab from the main menu.\n\n2.Select Nursing Home and Resident:\n\n-Choose the relevant Nursing Home from the dropdown menu.\n\n-Select the specific Resident whose medication stock you wish to audit.\n\n3.Download the Report:\n\n-Click on the PDF icon.\n\n-The stock audit report will open in PDF format, displaying detailed information.\n',
    "farewell": "Goodbye! Have a great day!",
    "complete_stock_audit_report":"Completing the stock audit for 25% of residents, including all medications such as short-term antibiotics and controlled drugs, helps ensure accurate tracking and accountability. This sampling method provides a reliable overview of stock management without auditing every resident.",
    "advantage_of_stock_audit_report":"The Stock Audit Report captures detailed data on medication stocks, usage, and discrepancies. It allows you to:\n\n•Monitor Stock Levels: Keep track of quantities carried over, received, and administered.\n\n•Identify Discrepancies: Compare calculated remaining quantities with actual stock.\n\n•Ensure Compliance: Maintain accurate records for regulatory requirements, especially for controlled substances.\n",
    "name":"Hi! Im Druground AI Assistance, Im here to help you, Feel free to ask",
    "acknowledgement":"Got it! How can I assist you further?",
    "apologies":"Sorry! Please forgive Me",
    "stock_report_details":"The Stock Audit report provides comprehensive details, including:\r\n\n•\tAppendix: Stock Count\r\n\n•\tDate Completed: [Date]\r\n\n•\tCompleted By: [Staff Member's Name]\r\n\nFields Included:\r\n\n•\tResident’s Initials\r\n\n•\tMedication\r\n\n•\tDose\r\n\n•\tQuantity Carried Over\r\n\n•\tQuantity Received\r\n\n•\tTotal Amount in Stock on Day 1 of Cycle\r\n\n•\tMiscellaneous Quantity\r\n\n•\tReturn Quantity\r\n\n•\tDisposed Quantity\r\n\n•\tQuantity Administered\r\n\n•\tQuantity Recorded as Not Given (for regular medicines)\r\n\n•\tCalculated Quantity Remaining\r\n\n•\tActual Quantity in Stock\r\n\n•\tCorrect or Incorrect",
}



def chatbot_response(query):
    # Step 1: Vectorize the query
    query_vec = vectorizer.transform([query])
    
    # Step 2: Predict the intent
    predicted_intent = model.predict(query_vec)[0]
    
    # Step 3: Return the response
    return responses.get(predicted_intent, "I'm sorry, I don't understand that. Can you rephrase?")




# print("Chatbot: Hi! Type 'quit' to exit.")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["quit", "exit"]:
#         print("Chatbot: Goodbye!")
#         break
#     response = chatbot_response(user_input)
#     print(f"Chatbot: {response}")
