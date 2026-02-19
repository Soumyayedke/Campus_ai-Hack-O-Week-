import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('punkt')

training_data = [
    ("How can I apply for admission?", "admission"),
    ("What is the admission process?", "admission"),
    ("When does enrollment start?", "admission"),
    ("How to join the college?", "admission"),

    ("When are the semester exams?", "exams"),
    ("Exam schedule for B.Tech?", "exams"),
    ("How are exams conducted?", "exams"),
    ("What is the exam pattern?", "exams"),

    ("What are the class timings?", "timetable"),
    ("Show me today's timetable", "timetable"),
    ("When do classes start?", "timetable"),
    ("Daily class schedule?", "timetable"),

    ("Is hostel facility available?", "hostel"),
    ("Hostel fees details?", "hostel"),
    ("Accommodation for girls?", "hostel"),
    ("Is mess included in hostel?", "hostel"),

    ("Do you offer scholarships?", "scholarships"),
    ("Scholarship eligibility criteria?", "scholarships"),
    ("How to apply for scholarship?", "scholarships"),
    ("Merit-based financial aid?", "scholarships"),

    ("What is the fee structure?", "fees"),
    ("How much is tuition?", "fees"),
    ("Payment details?", "fees"),
    ("Annual course cost?", "fees"),

    ("How are placements?", "placements"),
    ("Top recruiting companies?", "placements"),
    ("Placement percentage?", "placements"),
    ("Average salary package?", "placements"),
]

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

texts = [preprocess(item[0]) for item in training_data]
labels = [item[1] for item in training_data]

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

def predict_intent(user_query):
    user_query = preprocess(user_query)
    vector = vectorizer.transform([user_query])
    intent = classifier.predict(vector)[0]
    confidence = max(classifier.predict_proba(vector)[0])
    return intent, round(confidence, 2)

print("\nIntent Classifier (type 'exit' to quit)")
while True:
    query = input("\nEnter your query: ")
    if query.lower() == "exit":
        break
    
    intent, confidence = predict_intent(query)
    print("Predicted Intent:", intent)
    print("Confidence:", confidence)
