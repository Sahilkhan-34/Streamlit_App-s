# Streamlit Components 

import streamlit as st


st.header("This is Hearder")

st.date_input('DOB')


import pandas as pd
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

st.data_editor(data=df)

# picture = st.camera_input("Take A pic")
# if picture:
#     st.image(picture)

# st.form()

# with st.form("my_form"):
#     st.write("Inside the form")
#     slider_val = st.slider("Form slider")
#     checkbox_val = st.checkbox("Form checkbox")

#     # Every form must have a submit button.
#     submitted = st.form_submit_button("Submit")
#     if submitted:
#         st.write("slider", slider_val, "checkbox", checkbox_val)

# st.write("Outside the form")
# Inserting elements out of order:

# >>> import streamlit as st
# >>>
# >>> form = st.form("my_form")
# >>> form.slider("Inside the form")
# >>> st.slider("Outside the form")
# >>>
# >>> # Now add a submit button to the form:
# >>> form.form_submit_button("Submit")