import os
from pathlib import Path
import sys
import json

import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import streamlit as st

from optunizer.factory import parse_config


def main(**kwargs):
  st.set_page_config(page_title="Optunizer", page_icon="⚙️", layout="wide")
  url, study_name = None, None
  df = st.session_state['df'] if 'df' in st.session_state else None
  with st.sidebar:
    with st.expander('Data options'):
      with st.form(key='storage_params'):
        url = st.text_input('Storage URL', type='password').strip()
        url_button = st.form_submit_button(label='Update storage')
        if url_button:
          st.success('OK')
      if url:
        with st.form(key='study_params'):
          studies = optuna.study.get_all_study_summaries(storage=url)
          studies_names = [x.study_name for x in studies]
          study_name = st.selectbox(f'Study name', sorted(studies_names))
          study_button = st.form_submit_button(label='Load study')
          if study_button:
            df = load_study(url, study_name)
            st.session_state['df'] = df
            st.success('OK')
            stats = df['state'].value_counts().to_dict()
            stats['ROWS'] = len(df)
            stats['COLUMNS'] = len(df.columns)
            st.json(json.dumps(stats))
      if df is not None:
        csv = convert_df(df)
        st.download_button("Download CSV", csv, "data.csv", "text/csv") 
  if df is not None:
    tab_table, tab_scatter, tab_corr = st.tabs(['Table', 'Scatter', 'Correlation'])
    with st.sidebar:
      with st.expander('Layout options'):
        templates = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn",
                     "simple_white", "none"]
        template = st.selectbox('Template', templates, 
                                index=templates.index('plotly_dark'),
                                help='See also: ≡ → Settings → Theme')
        continuous_colors = px.colors.named_colorscales()
        continuous_color = st.selectbox('Continuous color', 
                                        continuous_colors, 
                                        index=continuous_colors.index('viridis'))
        is_continuous_reversed = st.checkbox('Reverse continuous', value=False)
        if is_continuous_reversed: 
          continuous_color += '_r'
        is_x_log = st.checkbox('Log X', value=False)
        is_y_log = st.checkbox('Log Y', value=False)
        is_y2_log = st.checkbox('Log Y2', value=False)
        x_title = st.text_input('Title X')
        y_title = st.text_input('Title Y')
        y2_title = st.text_input('Title Y2')
        legend_title = st.text_input('Title legend')
        colorbar_title = st.text_input('Title colorbar')
        fonts = ["Arial", "Balto", "Courier New", "Droid Sans", "Droid Serif", 
                 "Droid Sans Mono", "Gravitas One", "Old Standard TT", "Open Sans", 
                 "Overpass", "PT Sans Narrow", "Raleway", "Times New Roman", 
                 "Roboto", "Roboto Mono"]
        font_family = st.selectbox('Font', fonts, index=fonts.index('Roboto Mono'))
        font_size = st.number_input('Font size', min_value=0, value=12, step=1)
        layout = {'template': template, 'font_family': font_family, 'font_size': font_size}
        if is_x_log:  
          layout.setdefault('xaxis', {}).setdefault('type', 'log')
        if is_y_log:  
          layout.setdefault('yaxis', {}).setdefault('type', 'log')
        if is_y2_log:  
          layout.setdefault('yaxis2', {}).setdefault('type', 'log')
        if x_title:
          layout.setdefault('xaxis', {}).setdefault('title', {}).setdefault('text', x_title)
        if y_title:
          layout.setdefault('yaxis', {}).setdefault('title', {}).setdefault('text', y_title)
        if y2_title:
          layout.setdefault('yaxis2', {}).setdefault('title', {}).setdefault('text', y2_title)
        if legend_title:
          layout.setdefault('legend', {}).setdefault('title', {}).setdefault('text', legend_title)
        if colorbar_title:
          layout.setdefault('coloraxis', {}).setdefault('colorbar', {}).setdefault(
            'title', {}).setdefault('text', colorbar_title)
      with st.expander('Table options'):
        with st.form(key='table_params'):
          table_button = st.form_submit_button(label='Plot')
          if table_button:
            tab_table.dataframe(df)
            st.success('OK')
      with st.expander('Scatter options'):
        is_color = st.checkbox('Add color', value=True)
        is_symbol = st.checkbox('Add symbol', value=False)
        is_size = st.checkbox('Add size', value=False)
        with st.form(key='scatter_params'):
          x_col = st.selectbox(f'X', df.columns)
          y_col = st.selectbox(f'Y', df.columns)
          color_col = st.selectbox(f'Color', df.columns) if is_color else None
          symbol_col = st.selectbox(f'Symbol', df.columns) if is_symbol else None
          size_col = st.selectbox(f'Size', df.columns) if is_size else None
          hover_cols = st.multiselect('Hover', df.columns)
          do_add_params = st.checkbox('Add params to hover', value=False)
          scatter_button = st.form_submit_button(label='Plot')
          if scatter_button:
            if size_col is not None:
              df[size_col] = df[size_col].fillna(0)
            if symbol_col is not None:
              layout.setdefault('legend', {}).setdefault('x', 1.2)
            if do_add_params:
              params = {x for x in df.columns if x.startswith('params_')}
              params.update(hover_cols)
              hover_cols = list(params)
            fig = px.scatter(data_frame=df, x=x_col, y=y_col,
                             color=color_col, symbol=symbol_col, size=size_col,
                             hover_data=sorted(hover_cols),
                             color_continuous_scale=continuous_color)
            fig.update_layout(**layout)
            tab_scatter.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme=None)
            st.success('OK')
      with st.expander('Correlation options'):
        with st.form(key='correlation_params'):
          methods = ['pearson', 'kendall', 'spearman']
          method = st.selectbox(f'Method', methods, index=0)
          cell_size = st.number_input('Cell size', min_value=0, max_value=None, value=30, step=1)
          columns = st.multiselect('Columns', df.columns)
          do_add_params = st.checkbox('Add params to columns', value=False)
          corr_button = st.form_submit_button(label='Plot')
          if corr_button:
            if do_add_params:
              params = {x for x in df.columns if x.startswith('params_')}
              params.update(columns)
              columns = list(params)
            df2 = df[df['state'] == 'COMPLETE']
            corr = df2[columns].corr(method=method)
            fig = px.imshow(img=corr,
                            x=corr.columns, 
                            y=corr.columns, 
                            zmin=-1,
                            zmax=1,
                            color_continuous_scale=continuous_color, 
                            text_auto='.2f')
            # fig.update_traces(textfont={'size': 12})
            margin=dict(l=10, r=10, t=10, b=10)
            n_rows = len(corr.columns)
            n_cols = len(corr.columns)
            width = n_cols*cell_size + margin['l'] + margin['r']
            height = n_rows*cell_size + margin['t'] + margin['b']
            layout['width'] = width
            layout['height'] = height
            layout['margin'] = margin
            layout['coloraxis_showscale'] = False
            layout['autosize'] = False
            layout['plot_bgcolor'] = 'rgb(30, 30, 30)'
            layout.setdefault('xaxis', {}).setdefault('side', 'top')
            layout.setdefault('xaxis', {}).setdefault('visible', False)
            layout.setdefault('yaxis', {}).setdefault('visible', False)
            fig.update_layout(**layout)
            tab_corr.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme=None)
            st.success('OK')

            
def load_study(url, study_name):
  s = optuna.load_study(study_name=study_name, storage=url)
  df = s.trials_dataframe()
  if len(df[df['state'] == 'COMPLETE']) > 0:
    df["duration"] = df["duration"].dt.seconds
  return df
  

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


if __name__ == '__main__':
  c = None
  for i, a in enumerate(sys.argv):
    if 'script' == a:
      c = sys.argv[i + 1]
  if c is None:
    c = os.getenv('OPTUNA_CONFIG_APP', None)
  kwargs = parse_config(c)
  main(**kwargs)
