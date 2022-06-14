import packages.data_processor as dp
import streamlit as st 
import joblib

# Load the model
spam_clf = joblib.load(open('./models/log.pkl','rb'))

# Load vectorizer
vectorizer = joblib.load(open('./vectors/vectorizer.pickle', 'rb'))

### MAIN FUNCTION ###
def main(title = "Your Awesome Text classification App".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 65px; color: #4682B4;'>{}</h1>".format(title), 
    unsafe_allow_html=True)
    st.markdown(""" <style> .font {
                font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
                </style> """, unsafe_allow_html=True)
    st.markdown(""" <style> div[role=alert] {background-color: #F5EBFF} 
                </style> """, unsafe_allow_html=True)
    st.image("./images/2.jpg")
    info = ''
    
    with st.expander("1. Check the category of your text 😀"):
        text_message = st.text_input("Please enter your message")
        clean_text_message = dp.prepare_data(text_message)
        if st.button("Predict"):
            prediction = spam_clf.predict(vectorizer.transform([clean_text_message]))
            if(prediction[0] == 0):
                info = 'Electrical'
            elif(prediction[0] == 1):
                info = 'Gravitational'
            elif(prediction[0] == 2):
                info = 'Human'
            elif(prediction[0] == 3):
                info = 'Other'
            elif(prediction[0] == 4):
                info = 'Vehicular'
            else:
                info = 'Not recognized'
            st.info('Prediction: {}'.format(info))

if __name__ == "__main__":
    main()