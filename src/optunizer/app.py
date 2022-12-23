import os
from pathlib import Path
import sys
import json
import io

from matplotlib.tri import Triangulation, UniformTriRefiner, LinearTriInterpolator, CubicTriInterpolator
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from scipy.interpolate import griddata
import streamlit as st

from optunizer.factory import parse_config


def main(**kwargs):
  st.set_page_config(page_title="Optunizer", page_icon="⚙️", layout="wide")
  url, study_name = None, None
  df = st.session_state['df'] if 'df' in st.session_state else None
  df2 = st.session_state['df2'] if 'df2' in st.session_state else None
  fig_contour = st.session_state['fig_contour'] if 'fig_contour' in st.session_state else None
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
    tab_table, tab_scatter, tab_corr, tab_contour = st.tabs([
      'Table', 'Scatter', 'Correlation', 'Contour'])
    with st.sidebar:
      with st.expander('Transforms'):
        transforms_names = ['pandas', 'scale', 'post_pandas', 'filter']
        n_transforms = {}
        for t in transforms_names:
          n_transforms[t] = st.number_input(f'Number of {t}', min_value=0, max_value=None, value=0, step=1)
        with st.form(key='transforms'):
          transforms = []
          for name, number in n_transforms.items():
            for i in range(number):
              st.header(f'{name} {i+1}')
              if name in ['pandas', 'post_pandas']:
                if name == 'pandas':
                  base = st.selectbox('Base', df.columns, key=f'{name}_base_{i+1}')
                elif name == 'post_pandas':
                  base = st.text_input('Base', key=f'{name}_base_{i+1}')
                transform_kwargs = {
                  'func': st.text_input('Function', key=f'{name}_function_{i+1}'),
                  'base': base,
                  'new': st.text_input('New', key=f'{name}_new_{i+1}'),
                  'axis': st.selectbox('Axis', [None, 'rows', 'columns'], key=f'{name}_axis_{i+1}'),
                  'other': st.text_input('Other', key=f'{name}_other_{i+1}')}
              elif name == 'scale':
                transform_kwargs = {
                  'base': st.text_input('Base', key=f'{name}_base_{i+1}'),
                  'new': st.text_input('New', key=f'{name}_new_{i+1}'),
                  'initial': st.text_input('Initial', key=f'{name}_initial_{i+1}'),
                  'coefficient': st.text_input('Coefficient', key=f'{name}_coefficient_{i+1}')}
              elif name == 'filter':
                transform_kwargs = {
                  'func': st.selectbox('Function', ['ge', 'eq', 'lt', 'le', 'ne'], key=f'{name}_function_{i+1}'),
                  'base': st.selectbox('Base', df.columns, key=f'{name}_base_{i+1}'),
                  'other': st.text_input('Other', key=f'{name}_other_{i+1}')}
              else:
                raise NotImplemetedError(name)
              transform_kwargs = {k: v for k, v in transform_kwargs.items() 
                                  if v is not None and v != ''}
              for k, v in transform_kwargs.items():
                try:
                  transform_kwargs[k] = float(v)
                except:
                  pass
              transforms.append([name, transform_kwargs])
          transform_button = st.form_submit_button(label='Transform')
          if transform_button:
            df2 = df.copy(deep=True)
            n_rows, n_cols = len(df), len(df.columns)
            for transform_name, transform_kwargs in transforms:
              print(transform_name, transform_kwargs)
              if transform_name in globals():
                transform_function = globals()[transform_name]
                new_dfs = transform_function(df2, **transform_kwargs)
                df2 = pd.concat([df2] + new_dfs, axis=1)
              elif transform_name in ['pandas', 'post_pandas']:
                base = transform_kwargs.pop('base')
                new = transform_kwargs.pop('new')
                if 'other' in transform_kwargs:
                  if isinstance(transform_kwargs['other'], str):
                    transform_kwargs['other'] = df2[transform_kwargs['other']]
                df2[new] = df2[base].transform(**transform_kwargs)
              elif transform_name == 'filter':
                func = transform_kwargs.pop('func')
                base = transform_kwargs.pop('base')
                mask = getattr(df2[base], func)(**transform_kwargs)
                df2 = df2[mask]
              else:
                raise NotImplemetedError(name)
            st.session_state['df2'] = df2
            st.success(f'OK, ROWS: {n_rows}->{len(df2)}, COLS: {n_cols}->{len(df2.columns)}')
      if df2 is not None:
        df4 = df2
      else:
        df4 = df
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
        is_x_grid = st.checkbox('Grid X', value=False)
        is_y_grid = st.checkbox('Grid Y', value=False)
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
        layout.setdefault('xaxis', {}).setdefault('showgrid', is_x_grid)
        layout.setdefault('yaxis', {}).setdefault('showgrid', is_y_grid)
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
            tab_table.dataframe(df4)
            st.success('OK')
      with st.expander('Scatter options'):
        is_color = st.checkbox('Add color', value=True)
        is_symbol = st.checkbox('Add symbol', value=False)
        is_size = st.checkbox('Add size', value=False)
        with st.form(key='scatter_params'):
          x_col = st.selectbox(f'X', df4.columns)
          y_col = st.selectbox(f'Y', df4.columns)
          color_col = st.selectbox(f'Color', df4.columns) if is_color else None
          symbol_col = st.selectbox(f'Symbol', df4.columns) if is_symbol else None
          size_col = st.selectbox(f'Size', df4.columns) if is_size else None
          hover_cols = st.multiselect('Hover', df4.columns)
          do_add_params = st.checkbox('Add params to hover', value=False)
          scatter_button = st.form_submit_button(label='Plot')
          if scatter_button:
            if size_col is not None:
              df4[size_col] = df4[size_col].fillna(0)
            if symbol_col is not None:
              layout.setdefault('legend', {}).setdefault('x', 1.2)
            if do_add_params:
              params = {x for x in df4.columns if x.startswith('params_')}
              params.update(hover_cols)
              hover_cols = list(params)
            fig = px.scatter(data_frame=df4, x=x_col, y=y_col,
                             color=color_col, symbol=symbol_col, size=size_col,
                             hover_data=sorted(hover_cols),
                             color_continuous_scale=continuous_color)
            fig.update_layout(**layout)
            tab_scatter.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme=None)
            st.success('OK')
      with st.expander('Contour options'):
        contour_type = st.selectbox('Type', ['levels', 'constraint'])
        is_contour_labels = st.checkbox('Show labels', value=True)
        is_contour_lines = st.checkbox('Show lines', value=True)
        is_contour_mid = st.checkbox('Set mid', value=False)
        html_bytes = None
        with st.form(key='contour_params'):
          x_col = st.selectbox(f'X', df4.columns)
          y_col = st.selectbox(f'Y', df4.columns)
          z_col = st.selectbox(f'Z', df4.columns)
          contour_coloring = st.selectbox('Coloring', ['fill', 'heatmap', 'lines', 'none'])
          contour_smoothing = st.number_input('Smoothing', min_value=0., max_value=1.3, value=1.)
          if contour_type == 'levels':
            contour_start = st.number_input('Start')
            contour_end = st.number_input('End')
            contour_size = st.number_input('Size')
            contour_value_0, contour_value_1, contour_operation = None, None, None
          else:  # constraint
            contour_operation = st.selectbox(
              'Operation', ['=', '<', '>=', '>', '<=', '[]', '()', '[)', '(]', '][', ')(', '](', ')['])
            contour_value_0 = st.number_input('Value 0')
            contour_value_1 = st.number_input('Value 1')
            contour_start, contour_end, contour_size = None, None, None
          if is_contour_labels:
            contour_label_size = st.number_input(
              'Label Size', min_value=0, max_value=None, value=12, step=1)
            contour_label_color = st.selectbox(
              'Label Color', ['white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta'])
            contour_label_format = st.text_input(
              'Label Format', value='.0f', help='See https://github.com/d3/d3-format/tree/v1.4.5#d3-format')
          contour_interp = st.selectbox('Interpolation', ['linear', 'nearest', 'cubic'])
          contour_min_x = st.number_input('Min X')
          contour_max_x = st.number_input('Max X')
          contour_num_x = st.number_input('Num X', min_value=2, value=11)
          contour_min_y = st.number_input('Min Y')
          contour_max_y = st.number_input('Max Y')
          contour_num_y = st.number_input('Num Y', min_value=2, value=11)
          contour_min_z = st.number_input('Min Z')
          contour_max_z = st.number_input('Max Z')
          contour_mid_z = st.number_input('Mid Z') if is_contour_mid else None
          countour_button = st.form_submit_button(label='Plot')
          if countour_button:
            df4 = df4[[x_col, y_col, z_col]].dropna()
            xs = df4[x_col].to_numpy(copy=True)
            ys = df4[y_col].to_numpy(copy=True)
            zs = df4[z_col].to_numpy(copy=True)
            if contour_min_x == contour_max_x == 0:
              contour_min_x, contour_max_x = np.nanmin(xs), np.nanmax(xs)
            if contour_min_y == contour_max_y == 0:
              contour_min_y, contour_max_y = np.nanmin(ys), np.nanmax(ys)
            xr = np.linspace(contour_min_x, contour_max_x, contour_num_x)
            yr = np.linspace(contour_min_y, contour_max_y, contour_num_y)
            xr, yr = np.meshgrid(xr, yr)
            Z = griddata((xs, ys), zs, (xr, yr), method=contour_interp)
            contours = {'type': contour_type, 
                        'operation': contour_operation,
                        'coloring': contour_coloring}
            if contour_type == 'levels':
              if contour_start == contour_end == 0:
                contours['start'] = np.nanmin(Z)
                contours['end'] = np.nanmax(Z)
                if contour_size == 0:
                  contours['size'] = (np.nanmax(Z) - np.nanmin(Z)) / 10
                else:
                  contours['size'] = contour_size
              else:
                contours['start'] = contour_start
                contours['end'] = contour_end
                contours['size'] = contour_size
            else:
              if contour_operation in ['=', '<', '>=', '>', '<=']:
                contours['value'] = contour_value_0
              else:
                contours['value'] = [contour_value_0, contour_value_1]   
            if is_contour_labels:
              contours.update({      
                'showlabels': is_contour_labels,
                'labelformat': contour_label_format,
                'labelfont': {
                  'size': contour_label_size,
                  'color': contour_label_color}})
            contours['showlines'] = is_contour_lines
            contour_kwargs = {
              'x': xr[0], 'y': yr[:, 0], 'z': Z, 
              'line_smoothing': contour_smoothing,
              'colorscale': continuous_color,
              'contours': contours
            }
            try:
              contour_kwargs['colorbar'] = {'title': layout['coloraxis']['colorbar']['title']['text']}
            except Exception as e:
              contour_kwargs['colorbar'] = {'title': z_col}
            if contour_mid_z is not None:
              contour_kwargs['zmid'] = contour_mid_z
            if contour_min_z == contour_max_z == 0:
              contour_kwargs['zauto'] = True
            else:
              contour_kwargs['zauto'] = False
              contour_kwargs['zmin'] = contour_min_z
              contour_kwargs['zmax'] = contour_max_z
            # print(contour_kwargs)
            fig = go.Figure(data=go.Contour(**contour_kwargs))
            layout.setdefault('xaxis', {}).setdefault('title', {}).setdefault('text', x_col)
            layout.setdefault('yaxis', {}).setdefault('title', {}).setdefault('text', y_col)
            fig.update_layout(**layout)
            tab_contour.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme=None)
            st.success('OK')
            buffer = io.StringIO()
            fig.write_html(buffer, include_plotlyjs='cdn')
            fig_contour = buffer.getvalue().encode()
            st.session_state['fig_contour'] = fig_contour
        if fig_contour is not None:
          st.download_button(label='Download plot', data=fig_contour, 
                             file_name='plot.html', mime='text/html')
      with st.expander('Correlation options'):
        with st.form(key='correlation_params'):
          methods = ['pearson', 'kendall', 'spearman']
          method = st.selectbox(f'Method', methods, index=0)
          cell_size = st.number_input('Cell size', min_value=0, max_value=None, value=30, step=1)
          columns = st.multiselect('Columns', df4.columns)
          do_add_params = st.checkbox('Add params to columns', value=False)
          corr_button = st.form_submit_button(label='Plot')
          if corr_button:
            if do_add_params:
              params = {x for x in df4.columns if x.startswith('params_')}
              params.update(columns)
              columns = list(params)
            df4 = df4[df4['state'] == 'COMPLETE']
            corr = df4[columns].corr(method=method)
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
      # with st.expander('Contour options'):
      #   contour_type = st.selectbox('Type', ['levels', 'constraint'])
        # is_contour_labels = st.checkbox('Show labels', value=True)
        # is_contour_lines = st.checkbox('Show lines', value=True)
        # is_contour_mid = st.checkbox('Set mid', value=False)
        # with st.form(key='contour_params'):
        #   x_col = st.selectbox(f'X', df4.columns)
        #   y_col = st.selectbox(f'Y', df4.columns)
        #   z_col = st.selectbox(f'Z', df4.columns)
        #   contour_coloring = st.selectbox('Coloring', ['fill', 'heatmap', 'lines', 'none'])
        #   contour_smoothing = st.number_input('Smoothing', min_value=0., max_value=1.3, value=1.)
        #   if contour_type == 'levels':
        #     contour_start = st.number_input('Start')
        #     contour_end = st.number_input('End')
        #     contour_size = st.number_input('Size')
        #     contour_value_0, contour_value_1, contour_operation = None, None, None
        #   else:  # constraint
        #     contour_operation = st.selectbox(
        #       'Operation', ['=', '<', '>=', '>', '<=', '[]', '()', '[)', '(]', '][', ')(', '](', ')['])
        #     contour_value_0 = st.number_input('Value 0')
        #     contour_value_1 = st.number_input('Value 1')
        #     contour_start, contour_end, contour_size = None, None, None
        #   if is_contour_labels:
        #     contour_label_size = st.number_input(
        #       'Label Size', min_value=0, max_value=None, value=12, step=1)
        #     contour_label_color = st.selectbox(
        #       'Label Color', ['white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta'])
        #     contour_label_format = st.text_input(
        #       'Label Format', value='.0f', help='See https://github.com/d3/d3-format/tree/v1.4.5#d3-format')
        #   contour_interp = st.selectbox('Interpolation', ['linear', 'nearest', 'cubic'])
        #   contour_min_x = st.number_input('Min X')
        #   contour_max_x = st.number_input('Max X')
        #   contour_num_x = st.number_input('Num X', min_value=2, value=11)
        #   contour_min_y = st.number_input('Min Y')
        #   contour_max_y = st.number_input('Max Y')
        #   contour_num_y = st.number_input('Num Y', min_value=2, value=11)
        #   contour_min_z = st.number_input('Min Z')
        #   contour_max_z = st.number_input('Max Z')
        #   contour_mid_z = st.number_input('Mid Z') if is_contour_mid else None
        #   countour_button = st.form_submit_button(label='Plot')
        #   if countour_button:
        #     df4 = df4[[x_col, y_col, z_col]].dropna()
        #     xs = df4[x_col].to_numpy(copy=True)
        #     ys = df4[y_col].to_numpy(copy=True)
        #     zs = df4[z_col].to_numpy(copy=True)
        #     if contour_min_x == contour_max_x == 0:
        #       contour_min_x, contour_max_x = np.nanmin(xs), np.nanmax(xs)
        #     if contour_min_y == contour_max_y == 0:
        #       contour_min_y, contour_max_y = np.nanmin(ys), np.nanmax(ys)
        #     xr = np.linspace(contour_min_x, contour_max_x, contour_num_x)
        #     yr = np.linspace(contour_min_y, contour_max_y, contour_num_y)
        #     xr, yr = np.meshgrid(xr, yr)
        #     Z = griddata((xs, ys), zs, (xr, yr), method=contour_interp)
        #     contours = {'type': contour_type, 
        #                 'operation': contour_operation,
        #                 'coloring': contour_coloring}
        #     if contour_type == 'levels':
        #       if contour_start == contour_end == 0:
        #         contours['start'] = np.nanmin(Z)
        #         contours['end'] = np.nanmax(Z)
        #         if contour_size == 0:
        #           contours['size'] = (np.nanmax(Z) - np.nanmin(Z)) / 10
        #         else:
        #           contours['size'] = contour_size
        #       else:
        #         contours['start'] = contour_start
        #         contours['end'] = contour_end
        #         contours['size'] = contour_size
        #     else:
        #       if contour_operation in ['=', '<', '>=', '>', '<=']:
        #         contours['value'] = contour_value_0
        #       else:
        #         contours['value'] = [contour_value_0, contour_value_1]   
        #     if is_contour_labels:
        #       contours.update({      
        #         'showlabels': is_contour_labels,
        #         'labelformat': contour_label_format,
        #         'labelfont': {
        #           'size': contour_label_size,
        #           'color': contour_label_color}})
        #     contours['showlines'] = is_contour_lines
        #     contour_kwargs = {
        #       'x': xr[0], 'y': yr[:, 0], 'z': Z, 
        #       'line_smoothing': contour_smoothing,
        #       'colorscale': continuous_color,
        #       'contours': contours
        #     }
        #     try:
        #       contour_kwargs['colorbar'] = {'title': layout['coloraxis']['colorbar']['title']['text']}
        #     except Exception as e:
        #       contour_kwargs['colorbar'] = {'title': z_col}
        #     if contour_mid_z is not None:
        #       contour_kwargs['zmid'] = contour_mid_z
        #     if contour_min_z == contour_max_z == 0:
        #       contour_kwargs['zauto'] = True
        #     else:
        #       contour_kwargs['zauto'] = False
        #       contour_kwargs['zmin'] = contour_min_z
        #       contour_kwargs['zmax'] = contour_max_z
        #     # print(contour_kwargs)
        
#             fig = go.Figure(data=go.Contour(**contour_kwargs))
#             layout.setdefault('xaxis', {}).setdefault('title', {}).setdefault('text', x_col)
#             layout.setdefault('yaxis', {}).setdefault('title', {}).setdefault('text', y_col)
#             fig.update_layout(**layout)
#             tab_contour.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme=None)
#             st.success('OK')

            
def load_study(url, study_name):
  s = optuna.load_study(study_name=study_name, storage=url)
  df = s.trials_dataframe()
  if len(df[df['state'] == 'COMPLETE']) > 0:
    df["duration"] = df["duration"].dt.seconds
  return df


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def scale(df, base, new, initial, coefficient):
  coefficient = df[coefficient] if isinstance(coefficient, str) else coefficient
  initial = df[initial] if isinstance(initial, str) else initial
  base = df[base] if isinstance(base, str) else base
  new_df = coefficient*(base - initial) + initial
  new_df.name = new
  return [new_df]
  
  
def contours(df):
  fig = go.Figure(data=go.Contour(**contour_kwargs))
  layout.setdefault('xaxis', {}).setdefault('title', {}).setdefault('text', x_col)
  layout.setdefault('yaxis', {}).setdefault('title', {}).setdefault('text', y_col)
  fig.update_layout(**layout)
  tab_contour.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme=None)
  st.success('OK')
  # def nb_dx(input_file):
  # pd.set_option('display.max_columns', None)
  # p = Path(input_file)
  # df = preprocess(input_file)
  # print(len(df))
  # df = df[df['state'] == "COMPLETE"]
  # print(len(df))
  # df = df[df['concept'] == 'borehole']
  # df = df[df['size.rock'] == 10]
  # df = df[df['nb'] < 10]
  # df = df[df['dx'] < 100]
  # # print(len(df))
  # cols = ['q.limit', 'q.limit.container', 't.max',
  #         't.ebs.max', 't.castiron.max', 't.rock.max', 't.fill.max', 't.hill.max']
  # # Quality weighted
  # groups = df.groupby(['dz.rock', 'dr', 'dh', 'dx', 'nb'], as_index=False)
  # a2f = {x: 'last' for x in df}
  # for c in cols:
  #   a2f[c] = lambda x: np.average(x, weights=df.loc[x.index, "quality"])
  #   # a2f[c] = 'max'
  # df2 = groups.agg(a2f)
  # df = df2
  # # Extrapolate
  # order = 1
  # weights = [1, 1, 1, 1, 1]
  # cols_id = ['dx', 'nb', 'dr', 'dz.rock', 'dh']
  # polys = {x: {y: np.poly1d(np.polyfit(df[x], df[y], order)) for y in cols} for x in cols_id}
  # dfs = [df]
  # for ids in product([10, 15, 20, 30, 50], [2, 5, 10], [0.5, 1], [50], [1, 2]):
  #   r = df[np.logical_and.reduce([df[x] == y for x, y in zip(cols_id, ids)])]
  #   if len(r) == 0:
  #     row = {x: y for x, y in zip(cols_id, ids)}
  #     for c in cols:
  #       values = []
  #       for k, v in zip(cols_id, ids):
  #         mask = np.logical_and.reduce([df[x] == y for x, y in zip(cols_id, ids) if x != k])
  #         if len(df[mask]) > 1:
  #           vv = np.poly1d(np.polyfit(df[mask][k], df[mask][c], order))(v)
  #         else:
  #           vv = polys[k][c](v)
  #         values.append(vv)
  #       row[c] = np.average(values, weights=weights)
  #     dfs.append(pd.DataFrame([row]))
  # df = pd.concat(dfs, ignore_index=True)
  # df2 = df
  # # cnt = 0
  # # cols = ['dz.rock', 'dr', 'dh', 'dx', 'nb']
  # # for name, g in df.groupby(cols, as_index=False):
  # #   cnt += 1
  # #   print(name, len(g))
  # #   fig = px.scatter(g,
  # #                    x="quality",
  # #                    y='t.max',
  # #                    # color='zone',
  # #                    # facet_col='zone',
  # #                    # facet_col_wrap=1,
  # #                    log_x=True, 
  # #                    # trendline="ols", trendline_options=dict(log_x=True),
  # #                    trendline="lowess",
  # #                    # trendline="ols",     
  # #                    # trendline_options=dict(log_x=True),
  # #                    trendline_options=dict(frac=0.67),
  # #                    # trendline_color_override='white',
  # #                    template='plotly_dark',
  # #                    )
  # #   mask = np.logical_and.reduce([df2[x] == y for x, y in zip(cols, name)])
  # #   wa = df2[mask]
  # #   fig.add_hline(y=wa['t.max'].item(), line_width=1, line_dash="dash", line_color="white")
  # #   fig.write_image(f'group_{name}.png')
  # #   fig.write_html(f'group_{name}.html')
  # #   if cnt == 100:
  # #     break
  # # print(df2.head(5))
  # groups = df2.groupby(['dz.rock', 'dr', 'dh'], as_index=False)
  # n_levels = 50
  # num = 100
  # ms = 1
  # for name, g in groups:
  #   dz, dr, dh = name
  #   print(name, len(g))
  #   # # EBS
  #   # tri_by_xyz(p, g, 
  #   #            x='dx', y='nb', z='t.ebs.max', 
  #   #            x_label='Расстояние между скважинами, м', 
  #   #            y_label='Контейнеров на скважину, шт', 
  #   #            title= (f'Температура в ИББ при тепловыделении 1000 Вт/м3'
  #   #                    f'\nРасстояние между контейнерами {dh} м'
  #   #                    f'\nТолщина ИББ {dr} м' 
  #   #                    f'\nГлубина {dz} м'),
  #   #            cb_label='Температура, °C', suffix=f'ebs_max_{name}',
  #   #            colormap='jet', n_levels=n_levels, num=num, ms=ms)
  #   # # Rock
  #   # tri_by_xyz(p, g, x='dx', y='nb', z='t.rock.max', 
  #   #            x_label='Расстояние между скважинами, м', 
  #   #            y_label='Контейнеров на скважину, шт', 
  #   #            title= (f'Температура в среде при тепловыделении 1000 Вт/м3'
  #   #                    f'\nРасстояние между контейнерами {dh} м'
  #   #                    f'\nТолщина ИББ {dr} м' 
  #   #                    f'\nГлубина {dz} м'),
  #   #            cb_label='Температура, °C', suffix=f'rock_max_{name}',
  #   #            colormap='jet', n_levels=n_levels, num=num, ms=ms)
  #   # # EBS < 150
  #   # for k in [0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5]:
  #   #   t0 = 9.0
  #   #   c = 't.ebs.max'
  #   #   tlim = 150
  #   #   df2['dt'] = tlim - (k*(df2[c] - t0) + t0)
  #   #   tri_by_xyz_zero(p, g, x='dx', y='nb', z='dt', 
  #   #                   x_label='Расстояние между скважинами, м',
  #   #                   y_label='Контейнеров на скважину, шт', 
  #   #                   title= (f'Критерий 150°C в ИББ при тепловыделении {k*1000:.0f} Вт/м3'
  #   #                           f'\nРасстояние между контейнерами {dh} м'
  #   #                           f'\nТолщина ИББ {dr} м' 
  #   #                           f'\nГлубина {dz} м'),
  #   #                   cb_label='Запас по температуре, °C', suffix=f'ebs_{name}_{k}',
  #   #                   colormap='RdBu', n_levels=n_levels, num=num, linewidth=1, ms=ms)
  #   # # Rock < 100
  #   # for k in [0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5]:
  #   #   c = 't.rock.max'
  #   #   t0 = 9.0
  #   #   tlim = 100
  #   #   df2['dt'] = tlim - (k*(df2[c] - t0) + t0)
  #   #   tri_by_xyz_zero(p, g, x='dx', y='nb', z='dt', 
  #   #                   x_label='Расстояние между скважинами, м', 
  #   #                   y_label='Контейнеров на скважину, шт', 
  #   #                   title= (f'Критерий 100°C в среде при тепловыделении {k*1000:.0f} Вт/м3'
  #   #                           f'\nРасстояние между контейнерами {dh} м'
  #   #                           f'\nТолщина ИББ {dr} м' 
  #   #                           f'\nГлубина {dz} м'),
  #   #                   cb_label='Запас по температуре, °C', suffix=f'rock_{name}_{k}',
  #   #                   colormap='RdBu', n_levels=n_levels, num=num, linewidth=1, ms=ms)
  #   # All
  #   t0 = 9.0
  #   for k in [0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5]:
  #     g['Порода < 100°C'] = 100 - (k*(g['t.rock.max'] - t0) + t0)
  #     g['ИББ < 150°C'] = 150 - (k*(g['t.ebs.max'] - t0) + t0)
  #     contours_zero(p, g, x='dx', y='nb', z=['Порода < 100°C', 'ИББ < 150°C'], 
  #                   suffix=f'{name}_{k}', 
  #                   x_label='Расстояние между скважинами, м', 
  #                   y_label='Контейнеров на скважину, шт', 
  #                   title= (f'Критерии при тепловыделении {k*1000:.0f} Вт/м3'
  #                           f'\nРасстояние между контейнерами {dh} м'
  #                           f'\nТолщина ИББ {dr} м' 
  #                            f'\nГлубина {dz} м'),
  #                   colormap='RdBu', n_levels=n_levels, num=num, linewidth=1, ms=ms)
  

if __name__ == '__main__':
  c = None
  for i, a in enumerate(sys.argv):
    if 'script' == a:
      c = sys.argv[i + 1]
  if c is None:
    c = os.getenv('OPTUNA_CONFIG_APP', None)
  kwargs = parse_config(c)
  main(**kwargs)
