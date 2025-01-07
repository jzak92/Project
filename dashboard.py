import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pycaret.classification import*

# Dummy credentials for instructors
valid_users = {"mehboobali": "123", "jehanzaib": "456"}

# Initialize session state for login if not already set
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.login_error = False  # Initialize login_error

# Function to handle login
def check_login():
    username = st.session_state.username
    password = st.session_state.password
    if username in valid_users and valid_users[username] == password:
        st.session_state.logged_in = True
        st.session_state.login_error = False
    else:
        st.session_state.logged_in = False
        st.session_state.login_error = True

# Login form - Only show if not logged in
if not st.session_state.logged_in:
    st.image("ides.png", width=120)
    st.markdown("<h1 style='text-align: center;'>Instructor Dashboard Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create form for login inputs
        with st.form(key="login_form"):
            # Input fields for username and password
            username = st.text_input("Username", key="username")
            password = st.text_input("Password", type="password", key="password")

            # Login button
            submit_button = st.form_submit_button("Login", on_click=check_login)

        # Display login error or success message
        if st.session_state.login_error:
            st.error("Invalid username or password.")
        elif st.session_state.logged_in:
            st.success("Login successful. Welcome!")

else:

## Main app content is shown only after login

    # Load the dataset
    data = pd.read_csv('Preprocessed_data.csv')
    # Load the models
    gbc_model = load_model('GradientBoostingClassifier1')
    lightgbm_model = load_model('LGBMClassifier2')
    rf_model = load_model('RandomForestClassifier3')

    # Streamlit app interface
    st.image("ides.png", width=120, )
    st.markdown("<h1 style='text-align: center; color: black;'>Instructor Dashboard for E-Learning Systems</h1>", unsafe_allow_html=True)

    # Sidebar for additional options
    st.sidebar.header("Choose Desired Option")
    # Action selection: Choose between Predict, View Student Data, etc.
    action = st.sidebar.radio("Select Action", 
                                ("View Students Data", 
                                "Predict Performance", 
                                "Plot Heatmap",
                                "Compare ML Models"))


    ## 1-View Student Data Section
    if action == "View Students Data":
        st.header("Search Students Data")
        
        ## To get multiple students data
        # Input fields for Student IDs and Code Modules
        student_ids = st.text_input("Enter Student IDs (comma separated)").split(',')
        student_ids = [id.strip() for id in student_ids if id.strip()]

        # Select multiple Code Modules
        code_modules = st.multiselect("Select Code Modules", options=data['Code_module'].unique())

        # Search for the student data when the user clicks the button
        if st.button("Search"):
            if student_ids and code_modules:
                # Filter the data based on multiple Student IDs and Code Modules
                student_data = data[(data['Student_id'].isin([int(id) for id in student_ids])) &
                            (data['Code_module'].isin(code_modules))]
        
                # Display the student's data if found
                if not student_data.empty:
                    st.subheader(f"Details for selected students in selected modules")
                    st.write(student_data)
                else:
                    st.error("No student found with the provided details.")
            else:
                st.error("Please enter at least one Student ID and select at least one Code Module.")

    # # Input fields for Student ID and Code Module
        # student_id = st.text_input("Enter Student ID")
        # code_module = st.selectbox("Select Code Module", options=data['Code_module'].unique())
        # # Search for the student data when the user clicks the button
        # if st.button("Search"):
        #     # Filter the data based on Student ID and Code Module
        #     student_data = data[(data['Student_id'] == int(student_id)) & (data['Code_module'] == code_module)]
        #     # Display the student's data if found
        #     if not student_data.empty:
        #         st.subheader(f"Details for Student ID: {student_id} in Module: {code_module}")
        #         st.write(student_data)
        #     else:
        #         st.error("No student found with the provided details.")

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

        # Apply label encoding to each categorical feature
        label_encoder = LabelEncoder()
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
        
        comparison_df = pd.read_csv('comparison_results.csv')
        # Filter the comparison
        comp_models = comparison_df.head(3)[['Model', 'Accuracy', 'F1', 'Prec.', 'Recall']]
        # Display the comparison as a table in Streamlit
        st.write(comp_models)
        # Visualize the model comparison results using a bar chart
        st.subheader("Bar Plot: Model Comparison by Accuracy")
        plt.figure(figsize=(10, 6))
        comp_models.set_index('Model')['Accuracy'].plot(kind='bar', color=['blue', 'green', 'red'])
        plt.ylabel('Accuracy')
        plt.title('Model Comparison: Accuracy')
        st.pyplot(plt)