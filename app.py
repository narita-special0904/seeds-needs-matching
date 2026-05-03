import streamlit as st
import logging

from matching.app_business_challenge import render_business_challenge
from matching.app_target_customer import render_target_customer

# ロガー
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(level=logging.INFO, format=fmt)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logger = logging.getLogger("【streamlit_ui】")

from dotenv import load_dotenv
load_dotenv(".env", override=True)

# Streamlit
st.set_page_config(
    page_title="Seeds Needs Matching System", page_icon="🔍", layout="wide"
)
#======================================================================
# 簡易ログイン画面
#======================================================================
# 複数アカウント（ID：パスワード）
USER_CREDENTIALS = {
    "testuser": "testuser123",
}

def login():
    st.title(" :blue[Seeds Needs Matching System]")
    st.write("🔐 ログイン")

    # ログインフォーム
    with st.form(key="login_form"):
        st.text_input("ユーザーID", key="username_input")
        st.text_input("パスワード", type="password", key="password_input")
        # ログインボタン（フォールバック用）
        login_btn = st.form_submit_button("ログイン")

    if login_btn:
        username = st.session_state.username_input
        password = st.session_state.password_input

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.current_user = username  # オプション：ログインユーザー名を保存
            st.success("ログイン成功")
            logger.info(f"【ログインユーザ】{st.session_state.current_user} ")
            st.rerun()
        else:
            st.error("メールアドレスまたはパスワードが違います。")

#======================================================================
# メイン
#======================================================================
def main():

    # ラジオボタンをサイドバーに配置
    mode = st.sidebar.radio("機能を選択してください", ["経営課題観点", "対象顧客観点"], index=0, key="main_mode_selector")

    # 注意書き追加
    st.sidebar.markdown("⚠️ :red[処理途中でラジオボタンを変更しないでください]")

    # ログアウトボタン設置
    col1, col2 = st.columns([8, 1])

    with col1:
        # import html
        st.markdown("## :blue[Seeds Needs Matching System]")

        # ログインユーザー名表示
        email_raw = st.session_state.get('current_user', 'ユーザー')
        st.caption(f"現在、{email_raw} でログイン中です。\n\n")

        #==============================================================
        # 処理の切り替え
        #==============================================================
        if mode == "対象顧客観点":
            render_target_customer()
        else:
            render_business_challenge()

    # ログアウト専用
    with col2:
        if st.button("ログアウト", key="logout_button"):
            st.session_state.authenticated = False
            st.session_state.current_user = ""
            st.rerun()

#======================================================================================
# ログイン対応
#======================================================================================
# セッション状態の初期化
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 認証⇒ main関数コール（未認証⇒ login関数コール)
if st.session_state.authenticated:
    main()
else:
    login()

