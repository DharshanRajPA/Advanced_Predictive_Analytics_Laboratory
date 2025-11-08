import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample Data
data = {
    "Email Subject": [
        "Special Offer Just for You!", "Issue with My Recent Order", "Your Feedback Matters",
        "Meeting Agenda", "Support Needed for Installation", "Limited Time Discount",
        "Invoice for Your Recent Purchase", "Thank You for Your Inquiry", "Complaint About Service",
        "New Features Announcement", "Account Activation Instructions", "Customer Support Needed",
        "Subscription Renewal Notice", "Feedback on Our New Product", "Security Alert",
        "Thank You for Your Feedback", "Special Promotion for You", "Question About Product",
        "Complaint About Billing", "Request for Assistance", "Monthly Newsletter",
        "Order Confirmation", "Customer Inquiry", "Service Interruption Notice",
        "Support Request: Technical Issue", "Welcome to Our Service", "Cancellation Confirmation",
        "Feedback on Website", "Account Access Issues", "New Updates Available",
        "Discount for Loyal Customers", "Product Return Process", "Positive Feedback",
        "Team Meeting Schedule", "Guidance on Software Usage", "Exclusive Deal Inside",
        "Payment Confirmation", "Inquiry About Warranty", "Service Quality Complaint",
        "Update Your Profile", "Technical Support Needed", "User Satisfaction Survey",
        "Order Shipped Notification", "General Inquiry", "Maintenance Notice",
        "Troubleshooting Guide", "Welcome Email", "Order Cancellation Request", "Product Review Request"
    ],
    "Email Content": [
        "Get 50% off on your next purchase. Click here to claim the offer!",
        "I have an issue with my recent order. Please assist.",
        "Your feedback is valuable to us. Please share your thoughts.",
        "Please find the meeting agenda attached for tomorrow's meeting.",
        "I need support for installing the new software I purchased.",
        "Enjoy a limited time discount of 25% on all products.",
        "Thank you for your purchase. Attached is your invoice.",
        "Thank you for your inquiry. We will get back to you shortly.",
        "I am not satisfied with the service provided. Please address my complaint.",
        "We are excited to announce new features in our product.",
        "Follow these instructions to activate your account.",
        "I am facing issues with my recent order and need support.",
        "Your subscription is about to expire. Renew now to continue the service.",
        "We would love to hear your feedback on our new product.",
        "We detected a security alert on your account. Please review.",
        "Thank you for your feedback. We appreciate your input.",
        "Take advantage of this special promotion for a limited time only.",
        "I have a question about the product specifications.",
        "There is an issue with my billing. Please help.",
        "I need assistance with accessing my account.",
        "Here is your monthly newsletter with updates and news.",
        "Your order has been confirmed. Here are the details.",
        "I have an inquiry regarding your services.",
        "We regret to inform you about a temporary service interruption.",
        "I am experiencing a technical issue and need support.",
        "Welcome to our service! Here are some tips to get started.",
        "Your cancellation has been processed. We are sorry to see you go.",
        "We would like to hear your feedback on our website.",
        "I am unable to access my account. Please assist.",
        "New updates are available. Check them out now.",
        "Exclusive discount for loyal customers. Don't miss out!",
        "Please provide details on how to return a product.",
        "I am happy with the service and wanted to share positive feedback.",
        "The team meeting is scheduled for next Monday at 10 AM.",
        "I need guidance on how to use the new software.",
        "Check out this exclusive deal inside. Limited time offer!",
        "Your payment has been confirmed. Thank you for your purchase.",
        "Can you provide information about the product warranty?",
        "I am not happy with the quality of service. Please address my complaint.",
        "Please update your profile to ensure accurate information.",
        "I need technical support for the new software.",
        "We would like to hear your feedback on your recent experience.",
        "Your order has been shipped. Here are the tracking details.",
        "I have a general inquiry about your services.",
        "We will be performing maintenance on our servers.",
        "Here is a guide for troubleshooting common issues.",
        "Welcome to our service! We're glad to have you.",
        "Please cancel my recent order. Thank you.",
        "Please provide a review of the product you purchased."
    ],
    "Category": [
        "Spam", "Support Request", "Feedback", "Customer Inquiry", "Support Request",
        "Spam", "Customer Inquiry", "Customer Inquiry", "Complaint", "Feedback",
        "Support Request", "Support Request", "Customer Inquiry", "Feedback", "Support Request",
        "Feedback", "Spam", "Customer Inquiry", "Complaint", "Support Request",
        "Feedback", "Customer Inquiry", "Customer Inquiry", "Support Request", "Support Request",
        "Customer Inquiry", "Customer Inquiry", "Feedback", "Support Request", "Customer Inquiry",
        "Spam", "Support Request", "Feedback", "Customer Inquiry", "Spam",
        "Customer Inquiry", "Customer Inquiry", "Complaint", "Support Request", "Support Request",
        "Feedback", "Customer Inquiry", "Customer Inquiry", "Support Request", "Support Request",
        "Customer Inquiry", "Support Request", "Feedback"
    ]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Combine Email Subject and Email Content for vectorization
df['Combined'] = df['Email Subject'] + " " + df['Email Content']

# Convert Category to numerical labels
label_mapping = {
    "Spam": 0,
    "Customer Inquiry": 1,
    "Complaint": 2,
    "Feedback": 3,
    "Support Request": 4
}
df['Category'] = df['Category'].map(label_mapping)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Combined'], df['Category'], test_size=0.3, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_mapping.keys())
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
