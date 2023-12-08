# Importando a biblioteca streamlit:
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from PIL import Image
from streamlit_option_menu import option_menu

# Carregando o data frame:
voos = pd.read_csv('voos_raw.csv')
voos_eda = pd.read_csv('voos_eda (2).csv')
voos_eda = voos_eda.drop(columns=['Unnamed: 0'])
voos_eda['Dia Semana'] = voos_eda['Dia Semana'].astype(str)

# Definindo as configurações do APP:
st.set_page_config (layout="wide")

st.markdown("""<h1 style='text-align: center; font-size: 64px; padding-bottom: 4%'>
    Análise, Qualidade e Governança de Dados
    </h1>""", unsafe_allow_html=True)

menu_principal = option_menu(
    menu_title=None,
    options=['Resumo', 'Qualidade','Análise','Regressão'],
    icons = ['house-fill', 'house-fill', 'filetype-csv','telephone-fill'],
    #menu_icon = 'menu-app',
    default_index = 0,
    orientation = 'horizontal')

if menu_principal == 'Resumo':
    st.markdown("""<h2 style='text-align: center; font-size: 48px; padding-bottom: 4%'>
    Resumo do trabalho</h2>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.4,0.4], gap = 'medium')
    with col1:
        image = Image.open('airplane.jpg')
        st.image(image, caption = 'Designed by Freepik')
    with col2:
        st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
        O atraso dos voos tem efeitos econômicos negativos para passageiros, companhias aéreas e aeroportos, 
        além de desempenhar um papel importante tanto nos lucros como nas perdas das companhias aéreas. </p>
        <p style='text-align: justify; font-size: 28px; font-weight: bold;'>De acordo com dados do Bureau of 
        Transportation Statistics (BTS), mais de 20% dos voos dos Estados Unidos sofreram atrasos durante 2018, 
        o que resultou num grave impacto econômico equivalente a 41 bilhões de dólares.</p>
        <p style='text-align: justify; font-size: 28px; font-weight: bold;'>As razões para estes atrasos variam muito,
        desde o congestionamento aéreo às condições meteorológicas, problemas mecânicos, dificuldades no embarque de 
        passageiros e simplesmente a incapacidade da companhia aérea em dar resposta à procura dada a sua capacidade, 
        gerando custos elevados tanto para as companhias aéreas como para os seus passageiros.</p>
        <p style='text-align: justify; font-size: 28px; font-weight: bold;'>Portanto, uma estimativa precisa do 
        atraso dos voos é crítica para as companhias aéreas porque os resultados podem ser aplicados para aumentar 
        a satisfação do cliente e as receitas das agências aéreas.</p>""", unsafe_allow_html=True)
    st.markdown("""<p style='text-align: center; font-size: 20px; font-weight: bold; padding-top: 3%'> 
    Trabalho desenvolvido por Ícaro Cazé</p>""", unsafe_allow_html=True)

if menu_principal == 'Qualidade':
    st.markdown("""<h2 style='text-align: center; font-size: 48px;'>
        Qualidade dos dados</h2>""", unsafe_allow_html=True)


    with st.container():
        st.markdown("""<h3 style='text-align: center; font-size: 36px; padding-bottom: 4%;'>
        Dados Nulos</h3>""", unsafe_allow_html=True)
        col3, col4 = st.columns([0.5,0.5], gap = 'large')
        with col3:
            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            Observou-se que as variáveis “DEPARTURE_TIME” (horário de partida), “DEPARTURE_DELAY” 
            (atraso na partida), “ARRIVAL_TIME” (horário de chegada) e “ARRIVAL_DELAY” (atraso na chegada) 
            apresentavam valores nulos.</p><p style='text-align: justify; font-size: 29px; font-weight: bold;'>
            Ao analisar os dados nulos de horário de partida, verifica-se que os mesmos é porque o voo não chegou
            a partir, isto é, foram cancelados. Como o escopo do trabalho diz respeito a prever o tempo de atraso 
            não se um voo será cancelado ou não, optou-se por excluir os registros nulos.""", unsafe_allow_html=True)
        with col4:     
            image2 = Image.open('sl01.png')
            st.image(image2, caption = 'Quantidade de dados nulos')
        st.markdown("""<style>div[data-testid="stHorizontalBlock"] {display: flex; align-items: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="element-container"] {display: flex; justify-content: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""<h3 style='text-align: center; font-size: 36px; padding-bottom: 4%;'>
        Horário no formato float</h3>""", unsafe_allow_html=True) 
        col15, col16 = st.columns([0.5,0.5], gap = 'large')
        with col15:
            image4 = Image.open('sl04.png')
            st.image(image4, caption = 'Quantidade de dados nulos')
        with col16:
            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            Outro problema observado na base diz respeito aos tipos de dados, isto é, as variáveis que representam 
            os horários estimados e real de chegada e os horários estimados e real de partida encontram-se em formato 
            numérico,sendo necessário proceder com a transformação dos dados.""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="stHorizontalBlock"] {display: flex; align-items: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="element-container"] {display: flex; justify-content: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)

    with st.container():
        st.markdown("""<h3 style='text-align: center; font-size: 36px; padding-bottom: 4%;'>
        Avaliação do conjunto de dados</h3>""", unsafe_allow_html=True)
        col17, col18 = st.columns([0.5,0.5], gap = 'large')
        with col17:
            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            Antes da etapa de higienização.""", unsafe_allow_html=True)
            st.write(voos)
        with col18:
            
            voos = voos[voos['ARRIVAL_TIME'].notna()]
            voos = voos[voos['ARRIVAL_DELAY'].notna()]
            voos['DEPARTURE_TIME'] = voos['DEPARTURE_TIME'].astype('int').astype('str').str.zfill(4)
            voos['SCHEDULED_DEPARTURE'] = voos['SCHEDULED_DEPARTURE'].astype('int').astype('str').str.zfill(4)
            voos['SCHEDULED_ARRIVAL'] = voos['SCHEDULED_ARRIVAL'].astype('int').astype('str').str.zfill(4)
            voos['ARRIVAL_TIME'] = voos['ARRIVAL_TIME'].astype('int').astype('str').str.zfill(4)
            def converter_tempo(tempo):
                time_string = str(tempo).split('.')[0].zfill(4)
                try:
                    hour = int(time_string[:-2]) 
                    minute = int(time_string[-2:])
                    if hour >= 24:
                        hour = hour % 24
                    return f"{hour:02d}:{minute:02d}"
                except ValueError: 
                    return None
            voos['SCHEDULED_DEPARTURE'] = voos['SCHEDULED_DEPARTURE'].apply(converter_tempo)
            voos['DEPARTURE_TIME'] = voos['DEPARTURE_TIME'].apply(converter_tempo)
            voos['SCHEDULED_ARRIVAL'] = voos['SCHEDULED_ARRIVAL'].apply(converter_tempo)
            voos['ARRIVAL_TIME'] = voos['ARRIVAL_TIME'].apply(converter_tempo)
            voos['Partida'] =voos['SCHEDULED_DEPARTURE'].str.replace(r'\D', '', regex=True).astype(int)
            voos['DEPARTURE_TIME'] = pd.to_datetime(voos['DEPARTURE_TIME'], format='%H:%M')
            voos['SCHEDULED_ARRIVAL'] = pd.to_datetime(voos['SCHEDULED_ARRIVAL'], format='%H:%M')
            voos['ARRIVAL_TIME'] = pd.to_datetime(voos['ARRIVAL_TIME'], format='%H:%M')
            voos['SCHEDULED_DEPARTURE'] = pd.to_datetime(voos['SCHEDULED_DEPARTURE'], format='%H:%M')
            voos['DEPARTURE_TIME'] = pd.to_datetime(voos['DEPARTURE_TIME'], format='%H:%M').dt.time
            voos['SCHEDULED_ARRIVAL'] = pd.to_datetime(voos['SCHEDULED_ARRIVAL'], format='%H:%M').dt.time
            voos['ARRIVAL_TIME'] = pd.to_datetime(voos['ARRIVAL_TIME'], format='%H:%M').dt.time
            voos['SCHEDULED_DEPARTURE'] = pd.to_datetime(voos['SCHEDULED_DEPARTURE'], format='%H:%M').dt.time
            voos['Estação'] = ['Primavera' if 2<x<6 else 'Verão' if 5<x<9 else 
            'Outono' if 8<x<12 else 'Inverno' for x in voos['MONTH']]
            voos['Partida'] = voos['Partida'].astype(int)
            voos['Periodo'] = ['Manhã' if 559<x<1200 else 'Tarde' if 1159<x<1800 else 
            'Noite' if 1759<x<2359 else 'Madrugada' for x in voos['Partida']]
            voos['Atraso Total'] = voos['DEPARTURE_DELAY']+ voos['ARRIVAL_DELAY']
            voos['DAY_OF_WEEK'] = voos['DAY_OF_WEEK'].astype(str) 

            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            Após da etapa de higienização.""", unsafe_allow_html=True)
            st.write(voos)
        st.markdown("""<style>div[data-testid="stHorizontalBlock"] {display: flex; align-items: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="element-container"] {display: flex; justify-content: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)

if menu_principal == 'Análise':
    st.markdown("""<h2 style='text-align: center; font-size: 48px;'>
    Análise Exploratória</h2>""", unsafe_allow_html=True) 
   
    with st.container():
        st.markdown("""<h3 style='text-align: center; font-size: 36px; padding-bottom: 4%'>
        Quantidade de voos e Milhas percorridas por aérea</h3>""", unsafe_allow_html=True)
        col5, col6 = st.columns([0.6,0.4], gap = 'large')

        with col5:
            fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)
            voos_aereas = voos_eda['Aérea'].value_counts().sort_values(ascending=False)
            distancia_percorrida = voos_eda.groupby('Aérea')['Distancia'].sum().sort_values(ascending=False)
            voos_aereas_plot = sns.barplot(y=voos_aereas.values, x=voos_aereas.index, palette="mako",ax=ax1)
            distancia_percorrida_plot = sns.barplot(y=distancia_percorrida.values, x=distancia_percorrida.index, palette="mako",ax=ax2)
            voos_aereas_plot.set_xticklabels(voos_aereas_plot.get_xticklabels(), rotation=45, ha='right')
            distancia_percorrida_plot.set_xticklabels(distancia_percorrida_plot.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig, use_container_width=True)
        st.markdown("""<style>div[data-testid="stHorizontalBlock"] {display: flex; align-items: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="element-container"] {display: flex; justify-content: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        with col6:
            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;padding-top: 4%;'> 
            Observa-se a partir da figura que não necessariamente a quantidade de voos realizados pelas 
            companhias é proporcional a quantidade de milhas percorridas, como o caso da Alaska Airlines, que 
            apesar de ter realizado 39.720 voos em 2015, aproximadamente ⅔ da concorrente Hawaiian Airlines, 
            percorreu aproximadamente 18 milhões de milhas a mais que a Hawaiian Airlines.</p>""", 
            unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="stHorizontalBlock"] {display: flex; align-items: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="element-container"] {display: flex; justify-content: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)

    with st.container():
        st.markdown("""<h3 style='text-align: center; font-size: 36px; padding-bottom: 4%; padding-top: 4%;'>
        Tempo de atraso por companhia aérea</h3>""", unsafe_allow_html=True)
        col11, col12 = st.columns([0.5,0.5], gap = 'large')
        with col11:  
            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            A figura ao lado ilustra o atraso nos voos em relação às companhias aéreas. A partir desta figura, 
            observa-se que talvez o problema dos atrasos não estejam relacionados somente à complexidade da malha 
            aérea, mas também a outros fatores como a capacidade de atender as demandas por parte das companhias.
            É o caso da Frontier Airlines Inc., que apesar de ter realizado 8,14% do total de voos em 2015, 
            apresentou o maior tempo acumulado de atraso.</p>""", unsafe_allow_html=True)
        with col12:
            image3 = Image.open('sl03.png')
            st.image(image3, caption = 'Tempo de atraso por companhia aérea')
        st.markdown("""<style>div[data-testid="stHorizontalBlock"] {display: flex; align-items: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="element-container"] {display: flex; justify-content: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)

    with st.container():
        voos_eda = voos_eda.drop(columns=['Atraso Total'])
        codificador = LabelEncoder()
        voos_eda['Aérea']= codificador.fit_transform(voos_eda['Aérea'].values)
        voos_eda['Origem']= codificador.fit_transform(voos_eda['Origem'].values)
        voos_eda['Destino'] = codificador.fit_transform(voos_eda['Destino'].values)
        voos_eda['Dia Semana'] = codificador.fit_transform(voos_eda['Dia Semana'].values)
        voos_eda['Periodo'] = codificador.fit_transform(voos_eda['Periodo'].values)
        voos_eda['Estação'] = codificador.fit_transform(voos_eda['Estação'].values)
        X = voos_eda.drop('Atraso Chegada',axis = 1)
        y = voos_eda['Atraso Chegada']
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 2)
        X_train.loc[:, ['Distancia']] = PowerTransformer().fit_transform(X_train[['Distancia']])
        X_test.loc[:, ['Distancia']] = PowerTransformer().fit_transform(X_test[['Distancia']])
        std_scl = StandardScaler()
        X_train = std_scl.fit_transform(X_train)
        X_test = std_scl.fit_transform(X_test)
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)   
        st.markdown("""<h3 style='text-align: center; font-size: 36px; padding-bottom: 4%; padding-top: 4%;'>
        Correlação entre as variáveis</h3>""", unsafe_allow_html=True)
        col13, col14 = st.columns([0.5,0.5], gap = 'large')
        with col13:
            fig4, ax6 = plt.subplots(figsize=(12, 6))
            ax6 = sns.heatmap(X_train.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True, ax=ax6)
            plt.xlabel('Variáveis [@1: Aéreas, @2: Distância, @3: Origem, @3: Destino, @4: Dia Semana, \n @5: Período, @6: Estação, @7: Atraso Partida, @8: Atraso Chegada]')
            st.pyplot(fig4, use_container_width=True)
        with col14:  
            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            Após a transformação com o PowerTransformer(), procedeu-se com o escalonamento do conjunto de treino.
            Em seguida, avaliou a correlação entre as variáveis a partir de um gráfico de heatmap de todas as 
            variáveis selecionadas para compor o modelo de regressão linear. A partir do heatmap, observa-se que as variáveis 
            dependentes possuem baixo grau de correlação entre si.</p>""", unsafe_allow_html=True)

        st.markdown("""<style>div[data-testid="stHorizontalBlock"] {display: flex; align-items: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="element-container"] {display: flex; justify-content: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)

if menu_principal == 'Regressão':
    voos_eda = voos_eda.drop(columns=['Atraso Total'])
    codificador = LabelEncoder()
    voos_eda['Aérea']= codificador.fit_transform(voos_eda['Aérea'].values)
    voos_eda['Origem']= codificador.fit_transform(voos_eda['Origem'].values)
    voos_eda['Destino'] = codificador.fit_transform(voos_eda['Destino'].values)
    voos_eda['Dia Semana'] = codificador.fit_transform(voos_eda['Dia Semana'].values)
    voos_eda['Periodo'] = codificador.fit_transform(voos_eda['Periodo'].values)
    voos_eda['Estação'] = codificador.fit_transform(voos_eda['Estação'].values)
    X = voos_eda.drop('Atraso Chegada',axis = 1)
    y = voos_eda['Atraso Chegada']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 2)
    X_train.loc[:, ['Distancia']] = PowerTransformer().fit_transform(X_train[['Distancia']])
    X_test.loc[:, ['Distancia']] = PowerTransformer().fit_transform(X_test[['Distancia']])
    std_scl = StandardScaler()
    X_train = std_scl.fit_transform(X_train)
    X_test = std_scl.fit_transform(X_test)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    st.markdown("""<h2 style='text-align: center; font-size: 48px;'>
    Regressão Linear</h2>""", unsafe_allow_html=True) 
    with st.container():
        st.markdown("""<h3 style='text-align: center; font-size: 36px; padding-bottom: 4%'>
        Resultados</h3>""", unsafe_allow_html=True)
        col7, col8 = st.columns([0.5,0.5], gap = 'large')
        with col7:
            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            Para desenvolver o modelo de regressão linear múltipla utilizou-se a função LinearRegression(), 
            pertencente à biblioteca do sklearn.</p>
            <p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            Observa-se que o modelo apresenta valores próximos dos valores reais, porém em sua grande maioria, 
            os valores previstos não aparentam demonstrar muita correlação.</p>
            <p style='text-align: justify; font-size: 29px; font-weight: bold;'> A partir dos resultados obtidos, 
            conclui-se que talvez a regressão linear não seja o melhor modelo para representar o problema abordado, 
            uma vez que as métricas não atingiram valores satisfatórios, como também os valores previstos não estão 
            condizentes com os valores reais.</p>""", unsafe_allow_html=True)
        with col8:
            fig2, ax = plt.subplots()
            regressao_plot = sns.regplot(x = y_test, y = y_pred)
            ax.set_ylabel('Valor previsto pelo modelo')
            ax.set_xlabel('Valor real')
            st.pyplot(fig2, use_container_width=True)
        st.markdown("""<h3 style='text-align: center; font-size: 36px; padding-bottom: 4%; padding-top: 4%'>
        Métricas de Qualidade</h3>""", unsafe_allow_html=True)
        
        col9, col10 = st.columns([0.5,0.5])
        with col9:
            image3 = Image.open('sl2.png')
            st.image(image3, caption = 'Métricas de qualidade do modelo')
        with col10:
            st.markdown("""<p style='text-align: justify; font-size: 29px; font-weight: bold;'> 
            Avaliando a raiz do erro quadrático médio (RMSE), pode-se afirmar que, em média, o modelo está contendo
            um erro de 196,68 minutos em relação aos valores reais. A partir do erro médio absoluto (MAE), conclui-se 
            que o modelo tende a subestimar ou superestimar o tempo de atraso em 9,55 minutos.</p>
            <p style='text-align: justify; font-size: 29px; font-weight: bold;'> Outra possibilidade que talvez justifique
            o modelo de regressão linear não ter se adequado tão bem para esse problema diz respeito a possibilidade das  
            variáveis com maior correlação com a variável alvo tenham sido descartadas ao longo do processo de 
            higienização e preparação dos dados. </p>""", unsafe_allow_html=True)


        st.markdown("""<style>div[data-testid="stHorizontalBlock"] {display: flex; align-items: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>div[data-testid="element-container"] {display: flex; justify-content: center;
        font-size: 22px;]}</style>""", unsafe_allow_html=True)
