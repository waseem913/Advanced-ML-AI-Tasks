import streamlit as st
from ticket_classifier import load_tickets, zero_shot_tag

# Streamlit app title
st.title("Auto Tagging Support Tickets")

# Load tickets from CSV
tickets = load_tickets()

# Select ticket to classify
ticket_choice = st.selectbox("Choose a ticket:", tickets["ticket_text"])

# Button to get predicted tags
if st.button("Get Tags"):
    tags = zero_shot_tag(ticket_choice)
    st.write("Top 3 predicted tags:", tags)
