import os
from pathlib import Path
import sys
import json
import io
from pprint import pprint

# from matplotlib.tri import Triangulation, UniformTriRefiner, LinearTriInterpolator, CubicTriInterpolator
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
  fig_scatter = st.session_state['fig_scatter'] if 'fig_scatter' in st.session_state else None
  fig_contour = st.session_state['fig_contour'] if 'fig_contour' in st.session_state else None
  fig_fenia = st.session_state['fig_fenia'] if 'fig_fenia' in st.session_state else None
  fig_corr = st.session_state['fig_corr'] if 'fig_corr' in st.session_state else None
  tab_fenia, tab_table, tab_scatter, tab_corr, tab_contour = st.tabs([
    'FENIA', 'Table', 'Scatter', 'Correlation', 'Contour'])
  template_state = st.session_state.get('template_state', {})
  with st.sidebar:
    with st.expander('Data'):
      with st.form(key='storage_form'):
        url = st.text_input('Storage URL', type='password').strip()
        url_button = st.form_submit_button(label='Update storage')
        if url_button:
          st.success('OK')
      if url:
        with st.form(key='study_form'):
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
    with st.expander('Template'):
      template_dir = 'templates'
      template_path = Path(template_dir)
      template_path.mkdir(parents=True, exist_ok=True)
      template_files = [p for p in template_path.iterdir() if p.is_file()]
      template_files_names = sorted([x.stem for x in template_files])
      template_reset = st.button(label='Reset')  
      if template_reset:
        template_state = {}
        st.session_state['template_state'] = template_state
        st.success(f'Template reset')
        # st.experimental_rerun()  # auto?
      with st.form(key='template_form_load'):
        template_load_name = st.selectbox('Name', template_files_names)
        template_load = st.form_submit_button(label='Load')
        if template_load:
          template_file = template_path / f'{template_load_name}.json'
          with open(template_file) as f:
            template_state = json.load(f)
          pprint(template_state)
          # Cannot be modified after the widget is instantiated
          # https://docs.streamlit.io/library/api-reference/session-state#caveats-and-limitations
          # st.session_state.update(template_state)  
          st.session_state['template_state'] = template_state
          st.success(f'Template "{template_load_name}" loaded')
          st.experimental_rerun()
      with st.form(key='template_form_save'):
        template_save_name = st.text_input('Name')
        template_save = st.form_submit_button(label='Save')
        if template_save:
          template_file = template_path / f'{template_save_name}.json'
          template_state = {k: v for k, v in st.session_state.items()
                            if isinstance(v, (str, int, float, bool, list))}
          with open(template_file, 'w') as f:
            json.dump(template_state, f, indent=2)
          pprint(template_state)
          st.success(f'Template "{template_save_name}" saved')
          st.experimental_rerun()
    with st.expander('Layout'):
      template = init_template(template_state, st.selectbox, label='Template',
                               key='layout_template', 
                               options=["plotly", "plotly_white", "plotly_dark", 
                                        "ggplot2", "seaborn", "simple_white", "none"],
                               help='See also: ≡ → Settings → Theme')
      continuous_color = init_template(template_state, st.selectbox, label='Template',
                                       key='layout_continuous_colors',
                                       options=px.colors.named_colorscales(), 
                                       default='viridis')
      is_continuous_reversed = init_template(template_state, st.checkbox, 
                                             key='layout_is_continuous_reversed',
                                             label='Reverse continuous', default=False)
      if is_continuous_reversed: 
        continuous_color += '_r'
      is_x_log = init_template(template_state, st.checkbox, label='Log X', 
                               key='layout_is_x_log', default=False)
      is_y_log = init_template(template_state, st.checkbox, label='Log Y', 
                               key='layout_is_y_log', default=False)
      is_y2_log = init_template(template_state, st.checkbox, label='Log Y2', 
                                key='layout_is_y2_log', default=False)
      is_x_grid = init_template(template_state, st.checkbox, label='Grid X', 
                                key='layout_is_x_grid', default=False)
      is_y_grid = init_template(template_state, st.checkbox, label='Grid Y', 
                                key='layout_is_y_grid', default=False)
      title = init_template(template_state, st.text_input, label='Title', 
                            key='layout_title', default='')
      x_title = init_template(template_state, st.text_input, label='Title X', 
                              key='layout_x_title', default='')
      y_title = init_template(template_state, st.text_input, label='Title Y', 
                              key='layout_y_title', default='')
      y2_title = init_template(template_state, st.text_input, label='Title Y2', 
                               key='layout_y2_title', default='')
      legend_title = init_template(template_state, st.text_input, label='Title legend', 
                              key='layout_legend_title', default='')
      colorbar_title = init_template(template_state, st.text_input, label='Title colorbar', 
                              key='layout_colorbar_title', default='')
      font_family = init_template(
        template_state, st.selectbox, label='Font', key='layout_font', default='Roboto Mono',
        options=["Arial", "Balto", "Courier New", "Droid Sans", "Droid Serif", 
                 "Droid Sans Mono", "Gravitas One", "Old Standard TT", "Open Sans", 
                 "Overpass", "PT Sans Narrow", "Raleway", "Times New Roman", 
                 "Roboto", "Roboto Mono"])
      font_size = init_template(template_state, st.number_input, label='Font size', 
                                key='layout_font_size', default=12, min_value=0, step=1)
      is_autosize = init_template(template_state, st.checkbox, label='Autosize', 
                                  key='layout_autosize', default=True)
      height = init_template(template_state, st.number_input, 
                             label=f'Height', key=f'layout_height', default=400,
                             min_value=0, max_value=None, step=1)
      width = init_template(template_state, st.number_input, 
                            label=f'Width', key=f'layout_width', default=800,
                            min_value=0, max_value=None, step=1)
      # margin=dict(l=10, r=10, t=10, b=10)
      layout = {'template': template, 'font_family': font_family, 'font_size': font_size, 
                'autosize': is_autosize, 'height': height, 'width': width}
      if is_x_log:  
        layout.setdefault('xaxis', {}).setdefault('type', 'log')
      if is_y_log:  
        layout.setdefault('yaxis', {}).setdefault('type', 'log')
      layout.setdefault('xaxis', {}).setdefault('showgrid', is_x_grid)
      layout.setdefault('yaxis', {}).setdefault('showgrid', is_y_grid)
      if is_y2_log:  
        layout.setdefault('yaxis2', {}).setdefault('type', 'log')
      if title:
        layout.setdefault('title', {}).setdefault('text', title)
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
      # pprint(layout)
    with st.expander('FENIA'):
      fenia_type = st.selectbox('Type', ['time_zones'])
      with st.form(key='fenia_form'):
        fenia_files = st.file_uploader("Files", accept_multiple_files=True)
        fenia_k = init_template(template_state, st.number_input, 
                                label=f'Coefficient', key=f'fenia_coefficient', 
                                default=1.0)
        fenia_vol = init_template(template_state, st.number_input, 
                                  label=f'Volume', key=f'fenia_volume', 
                                  default=0.0)
        fenia_init_t = init_template(template_state, st.number_input, 
                                     label=f'Initial Temperature', key=f'fenia_initial_temperature', 
                                     default=9.0)
        fenia_button = st.form_submit_button(label='Plot')
        if fenia_button:
          if fenia_type == 'time_zones':
            max_zone_time_ts = []
            zone_names = []
            zone_volumes = []
            for f in fenia_files:
              buffer = io.StringIO(f.getvalue().decode("utf-8"))
              string_data = buffer.read()
              if 'maxZoneTt' in f.name:
                start_line = 13
                end_line = -1
                for line in string_data.split('\n')[start_line:end_line]:
                  line = line.strip()
                  if line != '':
                    n_values, values = line.split('(')
                    n_values = int(n_values)
                    values = [float(x) for x in values[:-1].split()]
                    max_zone_time_ts.append(values)
              if 'zoneNames' in f.name:
                start_line = 13
                end_line = -2
                string_data = string_data.replace('(', ' ').replace(')', ' ').replace(';', ' ')
                string_data = ' '.join(x.strip() for x in string_data.split('\n')[start_line:end_line])
                zone_names = [x for x in string_data.split()
                              if x != '' and not x.isnumeric()]
              if 'zoneVol' in f.name:
                start_line = 13
                end_line = -2
                string_data = string_data.replace('(', ' ').replace(')', ' ').replace(';', ' ')
                string_data = ' '.join(x.strip() for x in string_data.split('\n')[start_line:end_line])
                for token in string_data.split():
                  try:
                    token = float(token)
                  except:
                    continue
                  else:
                    zone_volumes.append(token)
                zone_volumes = zone_volumes[1:]
            print(zone_names)
            print(zone_volumes)
            zone2vol = dict(zip(zone_names, zone_volumes))
            zone2index = dict(zip(zone_names, range(len(zone_names))))
            zone2init = {x: fenia_init_t for x in zone_names}
            zone2init['Filling'] = 110
            zone2init['CastIron'] = 110
            print(zone2vol)
            print(zone2index)
            print(zone2init)
            for zone, init in zone2init.items():
              max_zone_time_ts[0][1+zone2index[zone]] = init
            columns = ['time'] + zone_names
            df_fenia = pd.DataFrame.from_records(max_zone_time_ts, columns=columns, index='time')
            df_fenia.index /= (86400*365.25)
            print(df_fenia)
            if fenia_vol != 0:
              k_vol = fenia_vol / zone2vol['Filling']
            else:
              k_vol = 1
            k_all = k_vol*fenia_k
            for zone in zone_names:
              if zone != 'Environment':
                df_fenia[zone] = k_all*(df_fenia[zone] - fenia_init_t) + fenia_init_t
              else:
                df_fenia[zone] = k_vol*0.92556*(df_fenia[zone] - fenia_init_t) + fenia_init_t
            print(df_fenia)
            eng2rus = {'Filling': 'РАО', 
                       'Plate': 'Перегородки', 
                       'Environment': 'Среда', 
                       'CastIron': 'ТУК', 
                       'EBS': 'Бентонит', 
                       'Wall': 'Стенки'}
            df_fenia = df_fenia.rename(eng2rus, axis='columns')
            fig = px.line(data_frame=df_fenia)
            fig.update_layout(**layout)
            tab_fenia.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme=None)
            buffer = io.StringIO()
            fig.write_html(buffer, include_plotlyjs='cdn')
            fig_fenia = buffer.getvalue().encode()
            st.session_state['fig_fenia'] = fig_fenia
          st.success('OK')
      if fig_fenia is not None:
        st.download_button(label='Download', data=fig_fenia, 
                           file_name='fenia.html', mime='text/html')  
  if df is not None:
    with st.sidebar:
      with st.expander('Transform'):
        transforms_names = ['pandas', 'post_pandas', 'scale', 'post_post_pandas', 'filter', 'slice']
        transform_numbers = {}
        for t in transforms_names:
          transform_numbers[t] = init_template(template_state, st.number_input, 
                                               label=f'Number of {t}', 
                                               key=f'transform_{t}_number', default=0, 
                                               min_value=0, max_value=None, step=1)
        with st.form(key='transform_form'):
          transforms = []
          for name, number in transform_numbers.items():
            for i in range(number):
              st.header(f'{name} {i+1}')
              base_options = list(df.columns)
              base_key = f'transform_{name}_base_{i+1}'
              base_value = template_state.get(base_key, '')
              base_index = base_options.index(base_value) if base_value in base_options else 0
              func_key = f'transform_{name}_function_{i+1}'
              func_value = template_state.get(func_key, '')
              new_key = f'transform_{name}_new_{i+1}'
              new_value = template_state.get(new_key, '')
              other_key = f'transform_{name}_other_{i+1}'
              other_value = template_state.get(other_key, '')
              axis_key = f'transform_{name}_axis_{i+1}'
              axis_options = [None, 'rows', 'columns']
              axis_value = template_state.get(axis_key, '')
              axis_index = axis_options.index(axis_value) if axis_value in axis_options else 0
              if 'pandas' in name:
                if name == 'pandas':
                  base = st.selectbox('Base', base_options, key=base_key, index=base_index)
                else:
                  base = st.text_input('Base', key=base_key, value=base_value)
                transform_kwargs = {
                  'func': st.text_input('Function', key=func_key, value=func_value),
                  'base': base,
                  'new': st.text_input('New', key=new_key, value=new_value),
                  'axis': st.selectbox('Axis', axis_options, key=axis_key, index=axis_index),
                  'other': st.text_input('Other', key=other_key, value=other_value)}
              elif name == 'scale':
                init_key = f'transform_{name}_initial_{i+1}'
                init_value = template_state.get(init_key, '')
                coef_key = f'transform_{name}_coefficient_{i+1}'
                coef_value = template_state.get(coef_key, '')
                transform_kwargs = {
                  # 'base': st.text_input('Base', key=base_key, value=base_value),
                  'base': st.selectbox('Base', base_options, key=base_key, index=base_index),
                  'new': st.text_input('New', key=new_key, value=new_value),
                  'initial': st.text_input('Initial', key=init_key, value=init_value),
                  'coefficient': st.text_input('Coefficient', key=coef_key, value=coef_value)}
              elif name == 'slice':
                start_key = f'transform_{name}_start_{i+1}'
                start_value = template_state.get(start_key, '')
                stop_key = f'transform_{name}_stop_{i+1}'
                stop_value = template_state.get(stop_key, '')
                step_key = f'transform_{name}_step_{i+1}'
                step_value = template_state.get(step_key, '')
                start_new = st.text_input('Start', key=start_key, value=start_value)
                stop_new = st.text_input('Stop', key=stop_key, value=stop_value)
                step_new = st.text_input('Step', key=step_key, value=step_value)
                start_new = int(start_new) if start_new else None
                stop_new = int(stop_new) if stop_new else None
                step_new = int(step_new) if step_new else None
                transform_kwargs = {
                  # 'base': st.text_input('Base', key=base_key, value=base_value),
                  'base': st.selectbox('Base', base_options, key=base_key, index=base_index),
                  'new': st.text_input('New', key=new_key, value=new_value),
                  'start': start_new,
                  'stop': stop_new,
                  'step': step_new}
              elif name == 'filter':
                filter_options = ['ge', 'eq', 'lt', 'le', 'ne']
                filter_index = filter_options.index(func_value) if func_value in filter_options else 0
                transform_kwargs = {
                  'func': st.selectbox('Function', filter_options, key=func_key, index=filter_index),
                  'base': st.selectbox('Base', base_options, key=base_key, index=base_index),
                  'other': st.text_input('Other', key=other_key, value=other_value)}
              else:
                raise NotImplementedError(name)
              transform_kwargs = {k: v for k, v in transform_kwargs.items() 
                                  if v is not None and v != ''}
              for k, v in transform_kwargs.items():
                if isinstance(transform_kwargs[k], str):
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
              elif 'pandas' in transform_name:
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
              elif transform_name == 'slice':
                base = transform_kwargs.pop('base')
                new = transform_kwargs.pop('new')
                df2[new] = df2[base].str.slice(**transform_kwargs)
              else:
                raise NotImplementedError(name)
            st.session_state['df2'] = df2
            st.success(f'OK, ROWS: {n_rows}->{len(df2)}, COLS: {n_cols}->{len(df2.columns)}')
      df4 = df2 if df2 is not None else df
      with st.expander('Table'):
        with st.form(key='table_form'):
          table_button = st.form_submit_button(label='Plot')
          if table_button:
            tab_table.dataframe(df4)
            st.success('OK')
      with st.expander('Scatter'):
        is_color = st.checkbox('Add color', value=True)
        is_symbol = st.checkbox('Add symbol', value=False)
        is_size = st.checkbox('Add size', value=False)
        with st.form(key='scatter_form'):
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
            buffer = io.StringIO()
            fig.write_html(buffer, include_plotlyjs='cdn')
            fig_scatter = buffer.getvalue().encode()
            st.session_state['fig_scatter'] = fig_scatter
            st.success('OK')
        if fig_scatter is not None:
          st.download_button(label='Download', data=fig_scatter, 
                             file_name='scatter.html', mime='text/html')
      with st.expander('Contour'):
        contour_type = init_template(template_state, st.selectbox, label=f'Type',
                                     key=f'contour_type', options=['levels', 'constraint'])
        is_contour_labels = init_template(template_state, st.checkbox, 
                                          label='Show labels', default=True,
                                          key='contour_is_contour_labels')
        is_contour_lines = init_template(template_state, st.checkbox, 
                                         label='Show line', default=True,
                                         key='contour_is_contour_lines')
        is_contour_mid = init_template(template_state, st.checkbox, 
                                       label='Set mid', default=False,
                                       key='contour_is_contour_mid')
        with st.form(key='contour_form'):
          x_col = init_template(template_state, st.selectbox, label=f'X',
                                key=f'contour_col_x', options=list(df4.columns))
          y_col = init_template(template_state, st.selectbox, label=f'Y',
                                key=f'contour_col_y', options=list(df4.columns))
          z_col = init_template(template_state, st.selectbox, label=f'Z',
                                key=f'contour_col_z', options=list(df4.columns))
          contour_coloring = init_template(template_state, st.selectbox, 
                                           label=f'Coloring', key=f'contour_coloring', 
                                           options=['fill', 'heatmap', 'lines', 'none']) 
          contour_smoothing = init_template(template_state, st.number_input, 
                                            label=f'Smoothing', key=f'contour_smoothing', 
                                            min_value=0., max_value=1.3, default=1.) 
          if contour_type == 'levels':
            contour_start = init_template(template_state, st.number_input, 
                                          label=f'Start', key=f'contour_start')
            contour_end = init_template(template_state, st.number_input, 
                                        label=f'End', key=f'contour_end')
            contour_size = init_template(template_state, st.number_input, 
                                         label=f'Size', key=f'contour_size')
            contour_value_0, contour_value_1, contour_operation = None, None, None
          else:  # constraint
            contour_operation = init_template(
              template_state, st.selectbox, label=f'Operation', 
              key=f'contour_operation', 
              options=['=', '<', '>=', '>', '<=', 
                       '[]', '()', '[)', '(]', '][', ')(', '](', ')['])
            contour_value_0 = init_template(template_state, st.number_input, 
                                            label=f'Value 0', key=f'contour_value_0')
            contour_value_1 = init_template(template_state, st.number_input, 
                                            label=f'Value 1', key=f'contour_value_1')
            contour_start, contour_end, contour_size = None, None, None
          if is_contour_labels:
            contour_label_size = init_template(template_state, st.number_input, 
                                               label=f'Label Size', 
                                               key=f'contour_label_size',
                                               min_value=0, max_value=None, step=1,
                                               default=12)
            contour_label_color = init_template(
              template_state, st.selectbox, label=f'Label Color',
              key=f'contour_label_color', 
              options=['white', 'black', 'red', 'green', 
                       'blue', 'yellow', 'cyan', 'magenta'])
            contour_line_color = init_template(
              template_state, st.selectbox, label=f'Line Color',
              key=f'contour_line_color', 
              options=['white', 'black', 'red', 'green', 
                       'blue', 'yellow', 'cyan', 'magenta'])
            contour_line_width = init_template(template_state, st.number_input, 
                                               label=f'Line Width', 
                                               key=f'contour_line_width',
                                               min_value=0., max_value=None,
                                               default=0.5)
            contour_label_format = init_template(
              template_state, st.text_input, label='Label Format', 
              key='layout_label_format', default='.0f',
              help='See https://github.com/d3/d3-format/tree/v1.4.5#d3-format')
          contour_interp = init_template(
              template_state, st.selectbox, label=f'Interpolation',
              key=f'contour_interpolation', 
              options=['linear', 'nearest', 'cubic'])
          contour_min_x = init_template(template_state, st.number_input, 
                                        label=f'Min X', key=f'contour_min_x')
          contour_max_x = init_template(template_state, st.number_input, 
                                        label=f'Max X', key=f'contour_max_x')
          contour_num_x = init_template(template_state, st.number_input, 
                                        label=f'Num X', key=f'contour_num_x',
                                        min_value=2, step=1, default=11)
          contour_min_y = init_template(template_state, st.number_input, 
                                        label=f'Min Y', key=f'contour_min_y')
          contour_max_y = init_template(template_state, st.number_input, 
                                        label=f'Max Y', key=f'contour_max_y')
          contour_num_y = init_template(template_state, st.number_input, 
                                        label=f'Num Y', key=f'contour_num_y',
                                        min_value=2, step=1, default=11)
          contour_min_z = init_template(template_state, st.number_input, 
                                        label=f'Min Z', key=f'contour_min_z')
          contour_max_z = init_template(template_state, st.number_input, 
                                        label=f'Max Z', key=f'contour_max_z')
          if is_contour_mid:
            init_template(template_state, st.number_input, 
                          label=f'Mid Z', key=f'contour_mid_z')
          else:
            contour_mid_z = None
          contour_button = st.form_submit_button(label='Plot')
          if contour_button:
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
              'line': {
                  'color': contour_line_color,
                  'width': contour_line_width},
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
          st.download_button(label='Download', data=fig_contour, 
                             file_name='contour.html', mime='text/html')
      with st.expander('Correlation'):
        with st.form(key='correlation_form'):
          methods = ['pearson', 'kendall', 'spearman']
          method = st.selectbox(f'Method', methods, index=0)
          cell_size = st.number_input('Cell size', min_value=0, max_value=None, value=30, step=1)
          columns = st.multiselect('Columns', df4.columns)
          do_add_params = st.checkbox('Add params to columns', value=False)
          do_plot_titles = st.checkbox('Plot titles', value=False)
          titles = st.text_input('Titles', help='Divided by "|"').split('|')
          plot_bgcolor = st.text_input('Background color', value='rgb(30, 30, 30)')
          corr_button = st.form_submit_button(label='Plot')
          if corr_button:
            if do_add_params:
              params = {x for x in df4.columns if x.startswith('params_')}
              params.update(columns)
              columns = list(params)
            df4 = df4[df4['state'] == 'COMPLETE']
            if len(titles) == len(columns) and do_plot_titles:
              old2new = dict(zip(columns, titles))
              df4 = df4.rename(old2new, axis='columns')
              columns = titles
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
            layout['plot_bgcolor'] = plot_bgcolor
            layout.setdefault('xaxis', {}).setdefault('side', 'top')
            layout.setdefault('xaxis', {}).setdefault('visible', do_plot_titles)
            layout.setdefault('yaxis', {}).setdefault('visible', do_plot_titles)
            fig.update_layout(**layout)
            tab_corr.plotly_chart(fig, use_container_width=False, sharing="streamlit", theme=None)
            buffer = io.StringIO()
            fig.write_html(buffer, include_plotlyjs='cdn')
            fig_corr = buffer.getvalue().encode()
            st.session_state['fig_corr'] = fig_corr
            st.success('OK')
        if fig_corr is not None:
          st.download_button(label='Download', data=fig_corr, 
                             file_name='corr.html', mime='text/html')  

            
def load_study(url, study_name):
  s = optuna.load_study(study_name=study_name, storage=url)
  df = s.trials_dataframe()
  if len(df[df['state'] == 'COMPLETE']) > 0:
    df["duration"] = df["duration"].dt.seconds
  return df


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def init_template(template_state, widget, key, options=None, default=None, **kwargs):
  """Wrapper of streamlit widget for initialization from template state
  """
  if key in template_state:
    value = template_state[key]
  elif default is not None:
    value = default
  else:
    value = None
  if options is not None and value in options:
    index = options.index(value)
  else:
    index = 0
  if widget == st.checkbox:
    return widget(key=key, value=value, **kwargs)
  elif widget == st.text_input:
    return widget(key=key, value=value, **kwargs)
  elif widget == st.number_input:
    if value is not None:
      return widget(key=key, value=value, **kwargs)
    else:
      return widget(key=key, **kwargs)
  elif widget == st.selectbox:
    return widget(options=options, key=key, index=index, **kwargs)
  elif widget == st.multiselect:
    return widget(key=key, default=value, **kwargs)
  else:
    raise NotImplementedError(widget)
  
  
def scale(df, base, new, initial, coefficient):
  coefficient = df[coefficient] if isinstance(coefficient, str) else coefficient
  initial = df[initial] if isinstance(initial, str) else initial
  base = df[base] if isinstance(base, str) else base
  new_df = coefficient*(base - initial) + initial
  new_df.name = new
  return [new_df]


if __name__ == '__main__':
  c = None
  for i, a in enumerate(sys.argv):
    if 'script' == a:
      c = sys.argv[i + 1]
  if c is None:
    c = os.getenv('OPTUNA_CONFIG_APP', None)
  kwargs = parse_config(c)
  main(**kwargs)
