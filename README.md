💼 K10 Career Advisor & Salary Predictor
Welcome to K10 Career Advisor, an AI-powered web app that predicts your expected salary range based on your academic and experiential background — and recommends personalized technical and soft skills to help you reach or exceed your career goals.

Built with TensorFlow, LangChain, Streamlit, and Mistral LLM, this tool aims to bridge the gap between student profiles and practical career planning.

🚀 Features
🎯 Salary Prediction using a trained Neural Network model

🤖 Career Skill Advice powered by Mistral 7B via LangChain-Together

📊 Intuitive UI with dynamic inputs for:

Projects

Internships

Certifications

Soft Skills

High School %

University GPA

Field of Study

Target Salary

🧠 Context-aware suggestions (strengths and weaknesses)

🌐 Deployed using Streamlit for easy access and interaction

🧠 Model & Feature Engineering
Inputs:
Academic and extra-curricular metrics collected from the user

Feature Engineering Strategy:
Weighted Features: Applied non-linear weighting to:

Projects Completed

Internships

Certifications

Soft Skills Score

High School Percentage

University GPA

Interaction Terms:

GPA × Projects

GPA × Internships

One-Hot Encoding for Field of Study

Scaling:

Used StandardScaler from sklearn on all numeric features

Model:
Architecture: Feedforward Neural Network with 4 hidden layers

Loss: Mean Squared Error (MSE)

Metrics: Mean Absolute Error (MAE)

Optimizer: Adam

Early stopping applied

📦 Installation
bash
Copy
Edit
git clone https://github.com/KetanSrivastava05/Career_Advisor.git
cd career-advisor
pip install -r requirements.txt
🔑 Required Secrets
Set your Together API Key in app.py:

python
Copy
Edit
together_api_key="your-api-key-here"
▶️ Running the App
bash
Copy
Edit
streamlit run app.py
🖼️ Screenshots

🧰 Tech Stack
Frontend: Streamlit

Backend: TensorFlow, LangChain

LLM: Mistral-7B-Instruct (via Together API)

Model Saving: .keras for model, joblib for scaler

Environment: Python 3.10+

📁 Project Structure

Copy
Edit
career_advisor/
│
├── app.py                   # Streamlit frontend logic
├── salary_predictor_model.keras  # Trained NN model
├── scaler1.save             # Trained StandardScaler
├── requirements.txt         # Project dependencies
├── README.md                

