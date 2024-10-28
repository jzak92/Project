import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pycaret.classification import *

# Dummy credentials for instructors
valid_users = {"mehboobali": "123", "jehanzaib": "456"}

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Define the login function
def login(username, password):
    if username in valid_users and valid_users[username] == password:
        st.session_state['logged_in'] = True
        st.success(f"Welcome, {username}!")
    else:
        st.error("Invalid username or password.")

# If not logged in, show the login form
if not st.session_state['logged_in']:
    # Center the login form
    st.image("ides.png", width=120)
    st.markdown("<h1 style='text-align: center;'>Instructor Dashboard Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Create a login form
        with st.form(key='login_form'):
            # Login form fields
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            # Form submit button
            submit_button = st.form_submit_button(label="Login")
        
        # Check login credentials when form is submitted
        if submit_button:
            login(username, password)
else:
# Main app content is shown only after login

    st.cache_data.clear()

    # Load the dataset
    data = pd.read_csv('Preprocessed_data1b.csv')

    # Load the models
    gbc_model = load_model('GradientBoostingClassifier1')
    lightgbm_model = load_model('LGBMClassifier2')
    rf_model = load_model('RandomForestClassifier3')

    # Streamlit app interface
    st.image("ides.png", width=120, )
    st.markdown("<h1 style='text-align: center; color: black;'>Instructor Dashboard for E-Learning Systems</h1>", unsafe_allow_html=True)
    # st.title("Instructor Dashboard for E-Learning Systems")

    # Sidebar for additional options
    st.sidebar.header("Choose Desired Option")

    # Action selection: Choose between Predict, View Student Data, etc.
    action = st.sidebar.radio("Select Action", 
                                ("View Student Data", 
                                "Predict Performance", 
                                "Plot Heatmap",
                                "Compare ML Models"))


    ## 1-View Student Data Section
    if action == "View Student Data":
        st.header("Search Student Data")
        
        # Input fields for Student ID and Code Module
        student_id = st.text_input("Enter Student ID")
        code_module = st.selectbox("Select Code Module", options=data['Code_module'].unique())
        
        # Search for the student data when the user clicks the button
        if st.button("Search"):
            # Filter the data based on Student ID and Code Module
            student_data = data[(data['Student_id'] == int(student_id)) & (data['Code_module'] == code_module)]
            
            # Display the student's data if found
            if not student_data.empty:
                st.subheader(f"Details for Student ID: {student_id} in Module: {code_module}")
                st.write(student_data)
            else:
                st.error("No student found with the provided details.")


    ## 2-Collect input data for student details (only for prediction)
    elif action == "Predict Performance":
        st.header("Enter Student Details to Predict Performance")
        student_id = st.text_input("Student ID")
        code_module = st.selectbox("Code Module", options=['AAA', 'BBB', 'CCC', 'DDD', 'FFF'])
        assessment_score = st.slider("Assessment Score", min_value=0.0, max_value=200.0)
        student_sum_click = st.number_input("Total Clicks", min_value=0)
        region = st.selectbox("Region", options=['London', 'Ireland', 'East Midlands', 'Yorkshire', 'Scotland', 'East Anglian', 'North', 'North Western', 'South East', 'South', 'South West', 'Wales', 'West Midlands'])
        highest_education = st.selectbox("Highest Education", options=['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification'])
        
        # Sidebar model selection
        st.sidebar.header("Choose Model")
        model_option = st.sidebar.selectbox(
            'Which model would you like to use?',
            ('Gradient Boosting','LightGBM','Random Forest')
        )
        
        # Mapping model option to actual model
        if model_option == 'Gradient Boosting':
            model = gbc_model
        elif model_option == 'LightGBM':
            model = lightgbm_model
        else:
            model = rf_model

        # When the instructor clicks 'Predict', run the prediction
        if st.button("Predict Performance"):
            input_data = {
                'Student_id': student_id,
                'Code_module': code_module,
                'Assessment_score': assessment_score,
                'Student_sum_click': student_sum_click,
                'Region': region,
                'Highest_education': highest_education
            }
            # Call the prediction function
            df = pd.DataFrame([input_data])
            predictions = predict_model(model, data=df)
            result = predictions['prediction_label'][0]  # Get the predicted label (Fail, Pass, Distinction)
            # Display the prediction result
            st.success(f"The predicted student performance is: {result}")


    ## 3-Heatmap Section
    elif action == "Plot Heatmap":
        st.header("Heatmap of Feature Correlations")
        data = pd.read_csv('Preprocessed_data1b.csv')

        label_encoder = LabelEncoder()
        # Apply label encoding to each categorical feature
        data['Code_module'] = label_encoder.fit_transform(data['Code_module'])
        data['Region'] = label_encoder.fit_transform(data['Region'])
        data['Highest_education'] = label_encoder.fit_transform(data['Highest_education'])
        data['Student_final_result'] = label_encoder.fit_transform(data['Student_final_result'])

        # Plot the heatmap using seaborn
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)


    ## 4-Model Comparison Section
    elif action == "Compare ML Models":
        st.header("Comparison of Different Models")
        
        # Compare models using PyCaret
        clf = setup(data=data, 
                target='Student_final_result',  # Actual target column
                train_size=0.7,  # 70% training, 30% testing
                normalize=True,  # normalize the features
                session_id=42)  # session_id is for reproducibility
        comparison_results = compare_models(n_select=3)
        
        # Pull the comparison table
        comparison_df = pull()
        # Filter the comparison_df to keep only the top 3 models
        top_3_models = comparison_df.head(3)
        
        # Display the comparison as a table in Streamlit
        st.write(top_3_models)
        
        # Visualize the model comparison results using a bar chart
        st.subheader("Bar Plot: Model Comparison by Accuracy")
        plt.figure(figsize=(10, 6))
        top_3_models.set_index('Model')['Accuracy'].plot(kind='bar', color=['blue', 'green', 'red'])
        plt.ylabel('Accuracy')
        plt.title('Model Comparison: Accuracy')
        st.pyplot(plt)