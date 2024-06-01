from streamlit.components.v1 import html
import CoolProp.CoolProp as props
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import math


def main(T_in, T_ref, uf, W, H, L1, L2, D, ST, SL, NT, NL1, NL2, t_fin, N_fin1, N_fin2):

    # CÁLCULOS GEOMÉTRICOS GERAIS
    Af = W*H #[m²] < 0.02 m²
    S_fin1 = W/(N_fin1+1) #[m]
    S_fin2 = W/(N_fin2+1) #[m]

    # ------------------------------------------------------------------------------------------------------------------------
    # ITERAÇÃO PARA CÁLCULO DE Q E T_out
    def iter(T_in, N_fin, NL, L, Af=Af, T_ref=T_ref, uf=uf, W=W, H=H, D=D, ST=ST, SL=SL, NT=NT, t_fin=t_fin):
        A_tubes = math.pi*D*W*NT*NL
        A_fin = 2*(L*H*N_fin - 0.25*NT*NL*math.pi*D**(2))
        As = A_tubes + A_fin
        Ac = H*W - N_fin*H*t_fin - 2*D*(W-t_fin)
        F_fin = As/A_tubes
        Sigma = Ac/Af
        Betta = As/(Af*L)
        Dh = 4*Sigma/Betta

        # ITERAÇÃO
        T_out = (T_in + T_ref)/2
        error = 1
        while error > 1e-4:
            T_out_old = T_out

            # AVALIAÇÃO DAS PROPRIEDADES DO AR EM Tm
            rho = props.PropsSI('D', 'T', (T_in + T_out)/2, 'P', 101325, 'Air')
            cp = props.PropsSI('C', 'T', (T_in + T_out)/2, 'P', 101325, 'Air')
            k = props.PropsSI('L', 'T', (T_in + T_out)/2, 'P', 101325, 'Air')
            mu = props.PropsSI('V', 'T', (T_in + T_out)/2, 'P', 101325, 'Air')
            Pr = (cp*mu)/k

            # CÁLCULOS TÉRMICOS
            m = rho*Af*uf
            C = m*cp
            G = m/Ac
            Re = G*Dh/mu
            j = 0.5685*(Re**(-0.4446))*(F_fin**(-0.3824))
            h = j*G*cp/(Pr**(2/3))

            # CÁLCULO DE PERDAS
            f = 5.9051*(Re**(-0.2973))*(F_fin**(-0.7487))*((NL/2)**(-0.4379))
            P = f*0.5*rho*((uf/Sigma)**2)*As/Ac

            # CÁLCULO DA RESISTÊNCIA TÉRMICA COM EFICIÊNCIA DE ALETA
            Xm, Xl = ST/2, math.sqrt((ST/2)*(ST/2) + (SL/2)*(SL/2))
            R = 1.28*Xm/((D/2)*math.sqrt((Xl/Xm) - 0.3))
            phi = (R - 1)*(1 + 0.35*math.log(R))
            y = math.sqrt(2*h/(226*t_fin)) # Cond. Térmica do Al ~= 226 W/mK
            ef_fin = math.tanh(phi*y*D/2)/(phi*y*D/2)
            U = h*(A_tubes + ef_fin*A_fin)/As

            # CÁLCULOS DE PROJETO
            NTU = U*As/C
            E = 1 - math.exp(-NTU)
            Q = E*C*(T_in-T_ref)
            T_out = T_in - Q/C

            # RESIDUO
            error = abs(T_out - T_out_old)
        
        # VERIFICAÇÃO DA ENTROPIA GERADA
        Tm = T_ref - Q/(h*As)
        Theta = (T_in-T_out)/T_ref
        Ec = (uf/Sigma)/math.sqrt(cp*Tm)
        Ns_T = ((Theta**2)/NTU)
        Ns_P = (Ec**2)*f*NTU*(Pr**(2/3))/(2*j)
        Ns = Ns_T + Ns_P
        print(f'Vazão volumétrica: {m/rho}')
        return T_out, Q, P, NTU, E, Betta, Sigma, L, As, cp, Pr, j, f, h, Ns_T, Ns_P, Ns, ef_fin

    # ------------------------------------------------------------------------------------------------------------------------
    # ITERAÇÃO PARA OTIMIZAÇÃO DO PROJETO: CÁLCULO DE T_ref E As
    def optimization(Q, T_in, T_out, N_fin, Sigma, cp, Pr, j, f, h, T_ref=T_ref, uf=uf, D=D, W=W, H=H, SL=SL, NT=NT):
        Tm = (T_in + T_out)/2
        Ec = (uf/Sigma)/math.sqrt(cp*Tm)
        error = 1
        while error > 1e-4:
            T_ref_old = T_ref
            Theta = (T_in-T_out)/T_ref
            NTU = Theta*math.sqrt(2*j/f)/(Ec*(Pr**(1/3)))
            E = 1 - math.exp(-NTU)
            T_ref = T_in - (T_in-T_out)/E
            error = abs(T_ref - T_ref_old)
        Ns_T = ((Theta**2)/NTU)
        Ns_P = (Ec**2)*f*NTU*(Pr**(2/3))/(2*j)
        Ns = Ns_T + Ns_P
        As = Q/(h*(Tm - T_ref))
        return NTU, E, T_ref, As, Ns_T, Ns_P, Ns

    # ------------------------------------------------------------------------------------------------------------------------
    # PÓS-OTIMIZAÇÃO: CALCULAR As BASEADO EM UM T_ref SELECIONADO
    def post_optimization(Q, T_in, T_out, N_fin, Sigma, cp, Pr, j, f, h, T_ref, uf=uf, D=D, W=W, H=H, SL=SL, NT=NT):
        Tm = (T_in + T_out)/2
        Ec = (uf/Sigma)/math.sqrt(cp*Tm)
        Theta = (T_in-T_out)/T_ref
        NTU = Theta*math.sqrt(2*j/f)/(Ec*(Pr**(1/3)))
        E = 1 - math.exp(-NTU)
        Ns_T = ((Theta**2)/NTU)
        Ns_P = (Ec**2)*f*NTU*(Pr**(2/3))/(2*j)
        Ns = Ns_T + Ns_P
        As = Q/(h*(Tm - T_ref))
        return NTU, E, T_ref, As, Ns_T, Ns_P, Ns

    # ------------------------------------------------------------------------------------------------------------------------
    # EXECUÇÃO DAS FUNÇÕES
    T_out1, Q1, P1, NTU1, E1, Betta1, Sigma1, L1, As1, cp1, Pr1, j1, f1, h1, Ns_T1, Ns_P1, Ns1, ef_fin1 = iter(T_in=T_in, N_fin=N_fin1, NL=NL1, L=L1)
    T_out2, Q2, P2, NTU2, E2, Betta2, Sigma2, L2, As2, cp2, Pr2, j2, f2, h2, Ns_T2, Ns_P2, Ns2, ef_fin2 = iter(T_in=T_out1, N_fin=N_fin2, NL=NL2, L=L2)

    NTU_otm1, E_otm1, T_ref_otm1, As_otm1, Ns_T_otm1, Ns_P_otm1, Ns_otm1 = optimization(Q=Q1, T_in=T_in, T_out=T_out1, N_fin=N_fin1, Sigma=Sigma1, cp=cp1, Pr=Pr1, j=j1, f=f1, h=h1)
    NTU_otm2, E_otm2, T_ref_otm2, As_otm2, Ns_T_otm2, Ns_P_otm2, Ns_otm2 = optimization(Q=Q2, T_in=T_out1, T_out=T_out2, N_fin=N_fin2, Sigma=Sigma2, cp=cp2, Pr=Pr2, j=j2, f=f2, h=h2)

    NTU_potm1, E_potm1, T_ref_potm1, As_potm1, Ns_T_potm1, Ns_P_potm1, Ns_potm1 = post_optimization(Q=Q1, T_in=T_in, T_out=T_out1, N_fin=N_fin1, Sigma=Sigma1, cp=cp1, Pr=Pr1, j=j1, f=f1, h=h1, T_ref=T_ref_otm2)
    NTU_potm2, E_potm2, T_ref_potm2, As_potm2, Ns_T_potm2, Ns_P_potm2, Ns_potm2 = post_optimization(Q=Q2, T_in=T_out1, T_out=T_out2, N_fin=N_fin2, Sigma=Sigma2, cp=cp2, Pr=Pr2, j=j2, f=f2, h=h2, T_ref=T_ref_otm2)

    # ------------------------------------------------------------------------------------------------------------------------
    # AVALIAÇÃO DOS DADOS GEOMÉTRICOS GLOBAIS
    L = L1 + L2  #[m] < 0.195 m
    As = As1 + As2 #[m²]
    NL = NL1 + NL2
    Betta = As/(Af*L)

    # AVALIAÇÃO DOS DADOS TERMODINÂMICOS GLOBAIS
    Q = Q1 + Q2 #[W] ~ 100 W
    rho = props.PropsSI('D', 'T', (T_in + T_out2)/2, 'P', 101325, 'Air')
    cp = props.PropsSI('C', 'T', (T_in + T_out2)/2, 'P', 101325, 'Air')
    E = Q / (rho*cp*Af*uf*(T_in-T_ref))
    NTU = math.log(1/(1-E))

    # AVALIAÇÃO DOS DADOS FLUIDODINÂMICOS GLOBAIS
    P = P1 + P2 #[Pa]

    # AVALIAÇÃO DA ENTROPIA GLOBAL
    Ns_T = Ns_T1 + Ns_T2
    Ns_P = Ns_P1 + Ns_P2
    Ns = Ns1 + Ns2

    # AVALIAÇÃO DOS DADOS OTIMIZADOS GLOBAIS
    NTU_otm = NTU_otm1 + NTU_otm2
    E_otm = 1 - math.exp(-NTU_otm)
    Ns_T_otm = Ns_T_otm1 + Ns_T_otm2
    Ns_P_otm = Ns_P_otm1 + Ns_P_otm2
    Ns_otm = Ns_otm1 + Ns_otm2
    As_otm = As_otm1 + As_otm2

    # AVALIAÇÃO DOS DADOS PÓS-OTIMIZADOS GLOBAIS
    NTU_potm = NTU_potm1 + NTU_potm2
    E_potm = 1 - math.exp(-NTU_potm)
    Ns_T_potm = Ns_T_potm1 + Ns_T_potm2
    Ns_P_potm = Ns_P_potm1 + Ns_P_potm2
    Ns_potm = Ns_potm1 + Ns_potm2
    As_potm = As_potm1 + As_potm2
    
    # ------------------------------------------------------------------------------------------------------------------------
    # RESULTADOS DO PROJETO ORGANIZADOS
    geometric_values = pd.DataFrame(data={
            'Comprimento (L) [m]': [round(L, 3), round(L1, 3), round(L2, 3)],
            'Largura (W) [m]': [round(W, 3), round(W, 3), round(W, 3)],
            'Altura (H) [m]': [round(H, 3), round(H, 3), round(H, 3)],
            'Área da Face (Af) [m²]': [round(Af, 4), round(Af, 4), round(Af, 4)],
            'Área da Superfície (As) [m²]': [round(As, 4), round(As1, 4), round(As2, 4)],
            'Número de Tubos Transversais (NT)': [NT, NT, NT],
            'Número de Tubos Longitudinais (NL)': [NL, NL1, NL2],
            'Número de Aletas (N_fin)': [N_fin2, N_fin1, N_fin2],
            'Eficiência de Aletas (ef_fin)': ['---', round(ef_fin1, 3), round(ef_fin2, 3)],
            'Fator de Compacidade (β)': [round(Betta, 3), round(Betta1, 3), round(Betta2, 3)],
            'Razão de Áreas (Σ)': ['---', round(Sigma1, 3), round(Sigma2, 3)]}, index = ['Total', '1ª Parte', '2ª Parte']).T
        
    termodynamic_values = pd.DataFrame(data={
            'Temperatura de saída (T_out) [ºC]': [round(T_out2-273.15, 3), round(T_out1-273.15, 3), round(T_out2-273.15, 3)],
            'Efetividade (E)': [round(E, 3), round(E1, 3), round(E2, 3)],
            'Nº de Unidades de Transferência (NTU)': [round(NTU, 3), round(NTU1, 3), round(NTU2, 3)],
            'Taxa de troca de calor (Q) [W]': [round(Q, 3), round(Q1, 3), round(Q2, 3)],
            'Perda de Carga (ΔP) [Pa]': [round(P, 3), round(P1, 3), round(P2, 3)]}, index = ['Total', '1ª Parte', '2ª Parte']).T
        
    entropy_values = pd.DataFrame(data={
        'Entropia Térmica (Ns_T)': [round(Ns_T, 6), round(Ns_T1, 6), round(Ns_T2, 6)],
        'Entropia de Pressão (Ns_P)': [round(Ns_P, 6), round(Ns_P1, 6), round(Ns_P2, 6)],
        'Entropia Total (Ns)': [round(Ns, 6), round(Ns1, 6), round(Ns2, 6)]}, index = ['Total', '1ª Parte', '2ª Parte']).T

    optimize_values = pd.DataFrame(data={
        'Entropia Térmica (Ns_T_otm)': [round(Ns_T_otm, 6), round(Ns_T_otm1, 6), round(Ns_T_otm2, 6)],
        'Entropia de Pressão(Ns_P_otm)': [round(Ns_P_otm, 6), round(Ns_P_otm1, 6), round(Ns_P_otm2, 6)],
        'Entropia Total (Ns_otm)': [round(Ns_otm, 6), round(Ns_otm1, 6), round(Ns_otm2, 6)],
        'Nº de Unidades de Transferência (NTU)': [round(NTU_otm, 3), round(NTU_otm1, 3), round(NTU_otm2, 3)],
        'Efetividade (E)': [round(E_otm, 6), round(E_otm1, 6), round(E_otm2, 6)],
        'Temperatura da Superfície (T_ref) (Cº)': ['---', round(T_ref_otm1-273.15, 3), round(T_ref_otm2-273.15, 3)],
        'Área da Superfície (As_otm) [m²]': [round(As_otm, 5), round(As_otm1, 5), round(As_otm2, 5)]}, index = ['Total', '1ª Parte', '2ª Parte']).T

    post_optimize_values = pd.DataFrame(data={
        'Entropia Térmica (Ns_T_potm)': [round(Ns_T_potm, 6), round(Ns_T_potm1, 6), round(Ns_T_potm2, 6)],
        'Entropia de Pressão(Ns_P_potm)': [round(Ns_P_potm, 6), round(Ns_P_potm1, 6), round(Ns_P_potm2, 6)],
        'Entropia Total (Ns_potm)': [round(Ns_potm, 6), round(Ns_potm1, 6), round(Ns_potm2, 6)],
        'Nº de Unidades de Transferência (NTU)': [round(NTU_potm, 3), round(NTU_potm1, 3), round(NTU_potm2, 3)],
        'Efetividade (E)': [round(E_potm, 6), round(E_potm1, 6), round(E_potm2, 6)],
        'Temperatura da Superfície (T_ref) (Cº)': ['---', round(T_ref_potm1-273.15, 3), round(T_ref_potm2-273.15, 3)],
        'Área da Superfície (As_potm) [m²]': [round(As_potm, 5), round(As_potm1, 5), round(As_potm2, 5)]}, index = ['Total', '1ª Parte', '2ª Parte']).T

    return geometric_values, termodynamic_values, entropy_values, optimize_values, post_optimize_values


# ------------------------------------------------------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA PARA VISUALIZAÇÃO DE RESULTADOS
st.set_page_config(
    page_title="Projeto Evaporador",
    page_icon=":flame:",
    layout="wide",
    initial_sidebar_state="expanded")
st.write('Elaborado por Antonio Carlos e Zara Michel')
st.title("PROJETO EVAPORADOR")
st.header("INPUTS DO PROJETO")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write('---')
    st.write('INPUTS TERMICOS / FLUIDODINÂMICOS')
    T_in = st.number_input('Temperatura na entrada (T_in) [K]', value=-18+273.15, format="%.4f")
    T_ref = st.number_input('Temperatura da superfície (T_ref) [K]', value=-30+273.15, format="%.4f")
    uf = st.number_input('Velocidade na face (uf) [m/s]', value=0.9, format="%.4f")
with col2:
    st.write('---')
    st.write('INPUTS GEOMÉTRICOS DOS TUBOS')
    D = st.number_input('Diâmetro dos Tubos (D) [m]', value=0.0079, format="%.4f")
    ST = st.number_input('Espaçamento Transversal entre Tubos (ST) [m]', value=0.023, format="%.3f")
    SL = st.number_input('Espaçamento Longitudinal entre Tubos (SL) [m]', value=0.022, format="%.3f")
    NT = st.number_input('Nº de Tubos Transversais (NT)', value=2)
    NL1 = st.number_input('Nº de Tubos Longitudinais na 1ª Parte (NL1)', value=3)
    NL2 = st.number_input('Nº de Tubos Longitudinais na 2ª Parte (NL2)', value=5)
with col3:
    st.write('---')
    st.write('INPUTS GEOMÉTRICOS DAS ALETAS')
    t_fin = st.number_input('Espessura de Aleta (t_fin) [m]', value=0.000127, format="%.6f")
    N_fin1 = st.number_input('Nº de Aletas na 1ª Parte (N_fin1)', value=10)
    N_fin2 = st.number_input('Nº de Aletas na 2ª Parte (N_fin2)', value=2*N_fin1 + 1)
with col4:
    st.write('---')
    st.write('INPUTS GEOMÉTRICOS GERAIS')
    W = st.number_input('Largura (W) [m]', value=0.2, format="%.4f")
    H = st.number_input('Altura (H) [m]', value=0.1, format="%.4f")
    L1 = st.number_input('Comprimento na 1ª Parte (L) [m]', value=NL1*SL, format="%.4f")
    L2 = st.number_input('Comprimento na 2ª Parte (L) [m]', value=NL2*SL, format="%.4f")
    

if st.button('EXECUTAR'):
    geometric_values, termodynamic_values, entropy_values, optimize_values, post_optimize_values = main(T_in, T_ref, uf, W, H, L1, L2, D, ST, SL, NT, NL1, NL2, t_fin, N_fin1, N_fin2)
    st.header("RESULTADOS DO PROJETO")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write('---')
        st.write('RESULTADOS GEOMÉTRICOS:')
        st.write(geometric_values)  
    with col2:
        st.write('---')
        st.write('RESULTADOS TÉRMICOS:')
        st.write(termodynamic_values)
        st.write('RESULTADOS DE ENTROPIA:')
        st.write(entropy_values)
    with col3:
        st.write('---')
        st.write('ANÁLISE DE OTIMIZAÇÃO:')
        st.write(optimize_values)
        st.write('DADOS DE OTIMIZAÇÃO BASEADOS EM T_ref_otm2:')
        st.write(post_optimize_values)
