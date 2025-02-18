import streamlit as st
import os
import pickle
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Loan Prediction",layout="wide",page_icon="🧑‍⚕️")

personal_model = pickle.load(open("personal_model.pkl", 'rb'))

home_model = pickle.load(open("home_model.pkl", 'rb'))

business_model = pickle.load(open("Business_model.pkl", 'rb'))

                  # sidebar for navigation
with st.sidebar:
    selected = option_menu('PERSONALIZED SMART LOAN ELIGIBILITY ANALYZER',

                           ['House Loan Prediction',
                            'Personal Loan Prediction',
                            'Business Loan Prediction'],
                           menu_icon = 'bar-chart',
                           icons = ['house', 'credit-card', 'briefcase'],
                           default_index=0)                

# house loan Prediction Page
if selected == 'House Loan Prediction':

    # page title
    st.title('House Loan Eligibility Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.number_input('Gender (Male 1, Female 0):',min_value=0,max_value=1)

    with col2:
        Married = st.number_input('Married (Married 1,Not Married 0):',min_value=0,max_value=1)

    with col3:
        Dependents = st.number_input('Dependents:',min_value=0)

    with col1:
        Education = st.number_input('Education(Graduate 0,Not Graduate 1):',min_value=0,max_value=1)

    with col2:
        Self_Employed = st.number_input('Self Employed (Yes 1,No 0):',min_value=0,max_value=1)

    with col3:
        ApplicantIncome = st.number_input('Applicant Income:', min_value=0.0) 


    with col1:
        CoapplicantIncome = st.number_input('Coapplicant Income:',min_value=0.0)

    with col2:
        LoanAmount = st.number_input('Loan Amount:',min_value=0)
    
    with col3:
        Loan_amount_term = st.number_input('Loan Amount Term (in months):',min_value=0)

    with col1:
        Credit_History = st.number_input('Credit History (Good 1,Bad 0):',min_value=0,max_value=1)

    with col2:
        Property_Area = st.number_input('Property Area (Rural 0,Semiurban 1,Urban 2):',min_value=0,max_value=3)


    # code for Prediction
    house_prediction = ''

    # Creating a button for Prediction
    if st.button('House Loan Eligibility Result'):

        # Collecting user inputs into a list for prediction
        user_input = [
            Gender, Married, Dependents, Education, Self_Employed,
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_amount_term, Credit_History, Property_Area
        ]
        
        # Converting the input list into a format suitable for prediction (usually 2D array)
        user_input = [user_input]

        # Convert user_input into a 2D array (reshape into 1 row, n columns)
        user_input = np.array(user_input).reshape(1, -1)

        # Assuming you have a trained 'house_model' (replace this with your actual model)
        house_prediction_result = home_model.predict(user_input)

        # Based on prediction, show the result
        if house_prediction_result[0] == 1:
            house_prediction = 'The customer is eligible for house loan.'
            image_path = "approved.png"  
            st.image(image_path, width=300)
        else:
            image_path = "Reject.png"  
            st.image(image_path, width=300)
            house_prediction = 'The customer is not eligible for house loan.'

    st.success(house_prediction)

#Personal Loan Prediction Page
if selected == 'Personal Loan Prediction':

    # page title
    st.title('Personal Loan Status Prediction using Ensembling Models')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age:',min_value=0)

    with col2:
        yrs_experience = st.number_input('Years of Experience:',min_value=0)

    with col3:
        family_size = st.number_input('Family Size:',min_value=0)

    with col1:
        education_level = st.number_input('Education Level (Graduate 0,Advanced or Professional 1, Undergraduate 2):',min_value=0,max_value=2)

    with col2:
        income = st.number_input('Income:',min_value=0.0)

    with col3:
        mortgage_amt = st.number_input('Mortgage Amount:',min_value=0.0)

    with col1:
        credit_card_acct = st.number_input('Credit Card Account (Yes 1,No 0):',min_value=0,max_value=1)

    with col2:
        credit_card_spend = st.number_input('Credit Card Spend:',min_value=0.0)

    with col3:
        share_trading_acct = st.number_input('Share Trading Account (Yes 1,No 0):',min_value=0,max_value=1)

    with col1:
        fixed_deposit_acct = st.number_input('Fixed Deposit Account (Yes 1,No 0):',min_value=0,max_value=1)

    with col2:
        online_acct = st.number_input('Online Account (Yes 1,No 0):',min_value=0,max_value=1)

    personal_prediction = ''

    # Creating a button for Prediction
    if st.button('Customer Loan Eligibility Result'):

        # Collecting user inputs into a list for prediction
        user_input = [
            age, yrs_experience, family_size, education_level,
            income, mortgage_amt, credit_card_acct, credit_card_spend,
            share_trading_acct, fixed_deposit_acct, online_acct
        ]
        
        # Convert user_input into a 2D array (reshape into 1 row, n columns)
        user_input = np.array(user_input).reshape(1, -1)

        # Assuming you have a trained 'loan_model' (replace this with your actual model)
        loan_prediction_result = personal_model.predict(user_input)

        # Based on prediction, show the result
        if loan_prediction_result[0] == 1:
            personal_prediction = 'The customer is eligible for personal loan.'
            image_path = "approved.png"  
            st.image(image_path, width=300)
        else:
            image_path = "Reject.png"  
            st.image(image_path, width=300)
            personal_prediction = 'The customer is not eligible for personal loan.'

    st.success(personal_prediction)


# Business Loan Prediction 
if selected == "Business Loan Prediction":

    # page title
    st.title("Business Loan Eligibility Prediction using DL Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        no_of_dependents = st.number_input('No of Dependents:',min_value=0)

    with col2:
        education = st.number_input('Education (Graduate 0, Undergraduate 1):',min_value=0,max_value=1)

    with col3:
        self_employed = st.number_input('Self Employed (No 0, Yes 1):',min_value=0,max_value=1)

    with col1:
        income_annum = st.number_input('Annual Income:',min_value=0.0)

    with col2:
        loan_amount = st.number_input('Loan Amount:',min_value=0.0)

    with col3:
        loan_term = st.number_input('Loan Term (in months):',min_value=0.0)

    with col1:
        cibil_score = st.number_input('Credit Score (usually between 300 and 900):',min_value=0)

    with col2:
        residential_assets_value = st.number_input('Residential Assets Value:',min_value=0)

    with col3:
        commercial_assets_value = st.number_input('Commercial Assets Value:',min_value=0)

    with col1:
        luxury_assets_value = st.number_input('Luxury Assets Value:',min_value=0)

    with col2:
        bank_asset_value = st.number_input('Bank Asset Value:',min_value=0)
   

   # code for Prediction
    business_loan_prediction = ''

    # creating a button for Prediction    
    if st.button('Business Loan Eligibility Result'):

        # Collecting user inputs into a list for prediction
        user_input = [
            no_of_dependents, education, self_employed, income_annum, loan_amount,
            loan_term, cibil_score, residential_assets_value, commercial_assets_value,
            luxury_assets_value, bank_asset_value
        ]
        
        # Convert user_input into a 2D array (reshape into 1 row, n columns)
        user_input = np.array(user_input).reshape(1, -1)

        # Assuming you have a trained 'business_loan_model' (replace this with your actual model)
        business_loan_prediction_result = business_model.predict(user_input)

        # Based on prediction, show the result
        if business_loan_prediction_result[0] == 1:
            business_loan_prediction = 'The customer is eligible for Business loan.'
            image_path = "approved.png"  
            st.image(image_path, width=300)
        else:
            image_path = "Reject.png"  
            st.image(image_path, width=300)
            business_loan_prediction = 'The customer is not eligible for Business loan.'

    st.success(business_loan_prediction)