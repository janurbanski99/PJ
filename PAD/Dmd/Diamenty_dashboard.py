import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf

def plot_scatter_and_line(x, scatter_y, line_y, scatter_name, line_name, title, x_title, y_title):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=scatter_y, name=scatter_name, mode="markers"))
    fig.add_trace(go.Scatter(
        x=x, y=line_y, name=line_name, line=dict(dash='dash', color='red')))
    fig.update_layout(title=title, xaxis_title=x_title,
        yaxis_title=y_title)
    return fig

all_cols = ['carat', 'clarity', 'color', 'cut', 'x_dimension', 'y_dimension', 'z_dimension', 'depth', 'table']

dmd_raw = pd.read_csv('dmd_raw.csv')
dmd = pd.read_csv('dmd.csv')
dmd.drop(columns='Unnamed: 0', inplace=True)

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>💎 Diamonds data dashboard 💎</h1>", unsafe_allow_html=True)
st.title('')
selected = option_menu(
    menu_title=None,
    options = ['Home', 'Heatmap', 'Distributions', 'Regression'],
    default_index=0,
    orientation='horizontal',
    icons = ['house', 'map', 'distribute-horizontal', 'arrow-up-right']
)

if selected == 'Home':
    st.markdown("<h5 style='text-align: center;'>Raw data overview</h5>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,5,2])
    col2.dataframe(dmd_raw.iloc[:, 1::].head(10))
    st.video("https://www.youtube.com/watch?v=vTfJp2Ts9X8")
elif selected == 'Heatmap':
    st.markdown("<h5 style='text-align: center;'>Correlation Heatmap</h5>", unsafe_allow_html=True)
    st.plotly_chart(px.imshow(dmd.corr()), theme=None, use_container_width=True)
elif selected == 'Distributions':
    var = st.selectbox(
   "Which variable distribution would you like to see?",
   ("carat", "x_dimension", "y_dimension", "z_dimension", "depth", "table", "price"),
   index=0,
   placeholder="Select variable...",
)
    n_bins = st.slider('Choose number of bins', min_value=1, max_value=30, step=1, value=15)
    st.plotly_chart(px.histogram(dmd, x=var, nbins=n_bins, title=f'Distribution of {var}'))
elif selected == "Regression":
    try:
        st.markdown("<h5 style='text-align: center;'>Custom Regression</h5>", unsafe_allow_html=True)
        vals = st.multiselect(
        'Choose your variables to visualize',
        ['carat', 'x_dimension', 'y_dimension', 'z_dimension', 'depth'], placeholder='Variables')
        fig = go.Figure()
        for i in vals:
            fig.add_trace(go.Scatter(x=dmd[i], y=dmd['price'], mode='markers', name=f'{i} vs price'))
            model = smf.ols(formula=f"price ~ {i}", data=dmd).fit()
            fig.add_trace(go.Scatter(x=dmd[i], y=model.fittedvalues, name=f"Fitted Regression Line for {i}"))
            fig.update_layout(title="Custom regression plot for price dependence on selected variables", 
                        yaxis_title='Price',
                        xaxis_title=" / ".join(vals))   
        # col1, col2, col3 = st.columns([1,6,1])
        st.plotly_chart(fig, use_container_width=True, use_container_height=True, width=400, height=500)
        try:
            # vals_reg = st.multiselect(
            # 'Choose your variables for custom regression model',
            # all_cols, placeholder='Variables', default=all_cols)

            # # st.write("Your formula: ", cust_formula)
            # st.latex(fr'Your \ formula: \ Price=\beta1*{vals_reg[0]} + \beta2*{vals_reg[1]} + \beta3*{vals_reg[2]} + \beta4*{vals_reg[3]} + \beta5*{vals_reg[4]} +\beta6*{vals_reg[5]} + \beta7*{vals_reg[6]} + \beta8*{vals_reg[7]} + \beta9*{vals_reg[8]} + \beta0')
            vals_reg = st.multiselect(
                'Choose your variables for custom regression model',
                all_cols, placeholder='Variables', default=all_cols)

            if vals_reg:
                start_formula = 'Price = '
                for index, val in enumerate(vals_reg[:-1]):
                    start_formula += f'\\beta_{index + 1} * \\verb|{val}| + '
                start_formula += f'\\beta_{len(vals_reg)} * \\verb|{vals_reg[-1]}|'
                st.latex(fr'Your formula: \ {start_formula} + \beta_0')
                start_formula_tofit = 'price ~ '
                for val in vals_reg:
                    start_formula_tofit += f'{val} + '
                cust_formula = start_formula_tofit[:-3]
            else:
                st.warning("Please select variables for the custom regression model.")                         
            dmd_cust_model = smf.ols(formula=cust_formula, data=dmd).fit()
            cust_coef = dmd_cust_model.params
            st.text(dmd_cust_model.summary())
        except:
            pass
        # st.latex(f"Yourcustom formula: ")
        # st.write(str(dmd.columns[]))
    except ValueError:
        pass
    st.markdown("<h5 style='text-align: center;'>Calculated Regression</h5>", unsafe_allow_html=True)
    dmd_model5 = smf.ols(formula='price ~ clarity + cut + x_dimension + y_dimension + table', data=dmd).fit()
    coefficients = dmd_model5.params
    variable_names = coefficients.index
    result_df = pd.DataFrame({'Variable': variable_names, 'Coefficient': coefficients})

    intercept_c = result_df.loc['Intercept'].Coefficient.round(2)
    clarity_c = result_df.loc['clarity'].Coefficient.round(2)
    cut_c = result_df.loc['cut'].Coefficient.round(2)
    x_dimension_c = result_df.loc['x_dimension'].Coefficient.round(2)
    y_dimension_c = result_df.loc['y_dimension'].Coefficient.round(2)
    table_c = result_df.loc['table'].Coefficient.round(2)
    st.latex(f"Price = {clarity_c} * Clarity {cut_c} * Cut + {x_dimension_c} * x\_dimension + {y_dimension_c} * y\_dimension + {table_c} * Table {intercept_c}")
    
    st.text(dmd_model5.summary())
    dmd['residuals'] = dmd_model5.resid
    
    line_y = [0] * len(dmd["price"])
    resid_plot = plot_scatter_and_line(dmd["price"], dmd["residuals"], line_y, "Model residuals", "y=0", "Model Residual Plot", "Price", "Residuals")
    st.plotly_chart(resid_plot, use_container_width=True, use_container_height=True, width=400, height=500)


