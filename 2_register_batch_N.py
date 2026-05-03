import os
import json
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import base64

import time

# 環境変数
from dotenv import load_dotenv
load_dotenv(".env", override=True)

# Azure関連
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from gremlin_python.driver import client, serializer
from gremlin_python.driver.protocol import GremlinServerError
from datetime import datetime, timedelta, timezone

# OpenAI関連
from openai import AzureOpenAI

# ログ
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("2_batch.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logger = logging.getLogger("【data_construction】")

class DataConstructionBatch:
    #----------------------------------------------------------------
    # 各種クライアント初期化
    #----------------------------------------------------------------
    def __init__(self):
        # バッチサイズ設定
        self.BATCH_SIZE = 10
        
        # APIレート制限対策の待機時間（秒）
        self.SLEEP_TIME_BETWEEN_BATCHES = 3
        
        # 登録失敗ドキュメントの記録（トランザクション管理用）
        self.failed_documents = {
            'ai_search': [],
            'gremlin_target_customer': [],
            'gremlin_business_challenge': []
        }
        self.successful_documents = {
            'ai_search': set(),
            'ai_search_gremlin_ids': set(),
            'gremlin_target_customer': set(),
            'gremlin_business_challenge': set()
        }
        
        # Azure Blob
        self.blob_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_BLOB_CONNECTION"))
        self.account_name = os.getenv("AZURE_BLOB_NAME")
        self.account_key = os.getenv("AZURE_BLOB_KEY")
        
        # Azure Document Intelligence
        self.doc_intelligence_client = DocumentIntelligenceClient(
            endpoint=os.getenv("AZURE_AI_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_AI_DOCUMENT_INTELLIGENCE_ENDPOINT_API_KEY"))
        )
        
        # Azure AI Search
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_AI_SEARCH_INDEX"),
            credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_ADMIN_KEY"))
        )
        
        # Azure ComosDB for Apache Gremlin
        self.gremlin_client_target_customer = client.Client(
            os.getenv("AZURE_COSMOSDB_GREMLIN_ENDPOINT"),
            'g',
            username=f"/dbs/{os.getenv('AZURE_COSMOSDB_GREMLIN_DB')}/colls/{os.getenv('AZURE_COSMOSDB_GREMLIN_CONTAINER_TARGET_CUSTOMER')}",
            password=os.getenv("AZURE_COSMOSDB_GREMLIN_READ_WRITE_KEY"),
            message_serializer=serializer.GraphSONSerializersV2d0()
        )

        self.gremlin_client_business_challenge = client.Client(
            os.getenv("AZURE_COSMOSDB_GREMLIN_ENDPOINT"),
            'g',
            username=f"/dbs/{os.getenv('AZURE_COSMOSDB_GREMLIN_DB')}/colls/{os.getenv('AZURE_COSMOSDB_GREMLIN_CONTAINER_BUSINESS_CHALLENGE')}",
            password=os.getenv("AZURE_COSMOSDB_GREMLIN_READ_WRITE_KEY"),
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        
        # Azure OpenAI
        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        
        # Azure Blob
        self.container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
        
        # メタタイプのマッピング
        self.meta_types = [
            "company_name", "location", "business_segment", "business_description", "value_proposition",
            "target_customer", "market_industry", "business_challenge", "social_issue",
            "technology_domain", "rnd_theme", "competitive_advantage"
        ]
        
        self.meta_type_mapping = {
            "企業名": "company_name",
            "所在地": "location",
            "事業セグメント": "business_segment",
            "事業内容": "business_description",
            "提供価値": "value_proposition",
            "対象顧客": "target_customer",
            "市場・業界": "market_industry",
            "経営課題": "business_challenge",
            "社会課題": "social_issue",
            "技術領域": "technology_domain",
            "研究開発テーマ": "rnd_theme",
            "競争優位性": "competitive_advantage"
        }

        # メタタイプノードのvalueにメタタイプの日本語をセットするための辞書作成
        self.meta_type_mapping_reverse = {v: k for k, v in self.meta_type_mapping.items()}


        # ベクトル次元定義(text-embedding-3-largeモデル)
        self.dimension = 3072
    
    # ------------------------------------------------------------
    #  安全な16桁 ID を生成
    # ------------------------------------------------------------
    def _to_safe_id(self, seed: str) -> str:
        h = hashlib.sha256(seed.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(h).decode()[:16]

    #----------------------------------------------------------------
    # BlobファイルからDocument Intelligenceでテキスト抽出
    #----------------------------------------------------------------
    def extract_text_from_blob(self, blob_path: str) -> str:
        try:
            blob_client = self.blob_client.get_blob_client(
                container=self.container_name, 
                blob=blob_path
            )
            blob_data = blob_client.download_blob().readall()
            
            # Azure AI Document Intelligence(Ver.1.0.20)でテキスト抽出
            try:
                poller = self.doc_intelligence_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=AnalyzeDocumentRequest(bytes_source=blob_data),
                    output_content_format="markdown",
                )
                
                result = poller.result()
                logger.info("✅️Azure AI Document Intelligence処理成功")
                return result.content

            except Exception as e:
                logger.error(f"❌️ ファイル「{blob_path}」：Azure AI Document Intelligence処理エラー：{e}")
                # エラーの場合にraiseで例外を再度スローすることで、外側のexceptに処理を伝播させる（2025/07/11)
                raise
        except Exception as e:
            logger.error(f"❌️テキスト抽出エラー {blob_path}: {e}")
            return ""

    #----------------------------------------------------------------------------
    # GPT-5.4で要約(content用)
    #----------------------------------------------------------------------------
    def summarize_by_gpt(self, content: str) -> str:
        if len(content) < 100:
            return content
        
        system_prompt = """あなたは優秀な文書要約の専門家です。
        以下の文書を要約するにあたり、次の点を守ってください：
        1. GPTモデルが後でユーザーの質問に正しく答えられるよう、前提情報・背景・文脈も含めてください。
        2. 内容の要点を簡潔かつ分かりやすく伝えてください。
        3. 3000文字以内に収めてください。
        4. **注意：『以下は提供された文書の要約です』のような出力はしないでください。**
        5. 本文のみを出力してください。
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content[:50000]}  # トークン制限を考慮
                ],
                max_completion_tokens=3000,
                # temperature=0.5  # 情報の正確性と自然な文章表現のバランスがとれた値を指定
            )
            logger.info("✅️要約(content用)成功")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌️要約(content用)エラー: {e}")
            return content[:3000]  # フォールバック

    #----------------------------------------------------------------------------
    # GPT-5.4で要約(content_for_embed用)
    #----------------------------------------------------------------------------
    def summarize_by_gpt_content_for_embed(self, content: str) -> str:
        if len(content) < 100:
            return content
        
        system_prompt = """あなたは優秀な文書要約の専門家です。
        以下の文書を要約するにあたり、次の点を守ってください：
        1. 重要な内容だけを抽出し、余分な説明や背景は省いてください。
        2. 600文字以内に収めてください。
        3. **注意：『以下は提供された文書の要約です』のような出力はしないでください。**
        4. 本文のみを出力してください。
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content[:50000]}  # トークン制限を考慮
                ],
                max_completion_tokens=800,
                # temperature=0.3  # ノイズの少ない安定したEmbeddingベクトル生成用として最適な値を指定
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌️要約(content_for_embed用)エラー: {e}")
            return content[:1000]  # フォールバック

    #----------------------------------------------------------------------------
    # 所与のテキストから3072次元ベクトル生成(text-embedding-3-largeモデル使用)
    #----------------------------------------------------------------------------
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = self.openai_client.embeddings.create(
                model=os.getenv("AZURE_OPENAI_MODEL_EMBEDDING_LARGE"),
                input=text
            )
            logger.info("✅️Embedding取得成功")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌️Embedding取得エラー: {e}")
            return [0.0] * self.dimension  # ALL0の3072次元ベクトル
    
    #----------------------------------------------------------------------------
    # メタ情報抽出（12種類のメタタイプ）※GPT-5.4使用
    #----------------------------------------------------------------------------
    def extract_meta_info(self, content: str) -> Dict[str, List[str]]:

        # チャンク分割
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "。", "、", " ", ""]
        )
        chunks = splitter.split_text(content)
        
        # メタ情報抽出
        summary = ""
        system_prompt = "あなたは文章を解析して重要なメタ情報を抽出するエキスパートです。"
        common_prompt = """### 制約条件 ###
 - 下記の出力形式は厳守してください。
 - 下記の出力形式の項目全てに対して出力してください。
 「企業名、所在地、事業セグメント、事業内容、提供価値、対象顧客、市場・業界、経営課題、社会課題、技術領域、研究開発テーマ、競争優位性」
 - 各項目に対する出力項目はMAX5件までを遵守してください。
 - 「顧客企業」と「業界」は、特に抽出結果が0件にならないよう広範囲での抽出を念頭に置いてください。
 - 抽出するメタ情報は体言止めでお願いします。句点「。」は最後に付加しないでください。
 - 該当項目のメタ情報が抽出出来なかった場合は何もセットしないでください。特に下記の出力形式例のような「xxx」をセットすることは厳禁です。

### 出力形式 ###
下記のように必ず半角カンマ「,」区切りで出力してください。全角カンマ「、」区切りは禁止です。	

企業名: xxx
所在地: xxx, xxx
...
競争優位性: xxx, xxx
"""

        for i, chunk in enumerate(chunks):
            if i == 0:
                prompt = f"""### 指示 ###
下記の12項目に対して、最大5件ずつ重要度の高い順にメタ情報を抽出してください。 メタ情報以外の説明などの出力は一切不要です。
「企業名、所在地、事業セグメント、事業内容、提供価値、対象顧客、市場・業界、経営課題、社会課題、技術領域、研究開発テーマ、競争優位性」

### 文章 ###
---
{chunk}
---

{common_prompt}
"""
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                prompt = f"""
以下の文章から、重要度の高い順に最大5件ずつ、前回までの抽出内容を考慮して、メタ情報が洗練されるように追加・更新してください。
それ以外の出力は一切不要です。
追加する場合は、*その追加メタ情報が既存のメタ情報より重要度が高い場合のみ*、重要度が一番低いメタ情報は除去した上で、
最大5件ずつになるように追加してください。

### 文章 ###
---
{chunk}
---

{common_prompt}
"""
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": summary}
                ]
            
            try:
                response = self.openai_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT"),
                    messages=messages,
                    max_completion_tokens=1500
                )
                summary = response.choices[0].message.content
                
                logger.info(f"📝 =======メタ情報抽出 (チャンク {i+1}/{len(chunks)})===============")
                logger.info(f"【抽出メタ情報】\n{summary}")
                logger.info("📝 ================================================================\n")
                
            except Exception as e:
                logger.error(f"チャンク {i} のメタ情報抽出エラー: {e}")
        
        # 結果をパースして辞書形式に変換
        result = {}
        if summary:
            for line in summary.strip().splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if not k or not v:
                    continue
                mapped_key = self.meta_type_mapping.get(k)
                if mapped_key:
                    # カンマ区切りで最大5つまで
                    items = [item.strip() for item in v.replace("、", ",").split(",") if item.strip()]
                    result[mapped_key] = items[:5]
        
        # 存在しないキーは空リストで初期化
        for meta_type in self.meta_types:
            if meta_type not in result:
                result[meta_type] = []
        
        return result
    
    #----------------------------------------------------------------------------
    # 対象課題メタタイプノードのベクトル生成
    #----------------------------------------------------------------------------
    def create_target_metainfo_vector(self, meta_infos: List[str]) -> List[float]:
        if not meta_infos:
            return [0.0] * self.dimension
        
        # メタ情報を結合（重要度順で重み付け）
        weighted_text = ""
        for i, info in enumerate(meta_infos[:5]):  # 最大5つ
            weight = 5 - i  # 重要度に応じた重み（メタ情報抽出時に「重要度順にMAX5出力」としているため
            weighted_text += f"{info} " * weight
        
        return self.get_embedding(weighted_text.strip())
    
    #----------------------------------------------------------------------------
    # ファイルパスの安全なID生成
    #----------------------------------------------------------------------------
    def create_safe_id(self, filepath: str) -> str:
        # ハッシュ化してBase64エンコード
        hash_obj = hashlib.sha256(filepath.encode('utf-8'))
        base64_hash = base64.urlsafe_b64encode(hash_obj.digest()).decode('utf-8')
        # 最初の16文字を使用
        return base64_hash[:16]
    
    #----------------------------------------------------------------------------
    # BlobファイルのSAS URLを生成
    #----------------------------------------------------------------------------
    def generate_blob_sas_url(self, blob_path: str) -> str:
        try:
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=blob_path,
                account_key=self.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(hours=24)  # 24時間有効
            )
            
            blob_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_path}?{sas_token}"
            return blob_url
        except Exception as e:
            logger.error(f"❌️SAS URL生成エラー: {e}")
            return ""

    #----------------------------------------------------------------------
    # Gremlinグラフの全頂点をタイムアウトせずに削除する関数
    #----------------------------------------------------------------------
    def _clear_graph_in_batches(self, _gremlin_client) -> None:

        BATCH_SIZE = 100
        labels = ["File", "MetaType", "MetaInfo"]

        for lab in labels:
            while True:
                ids = _gremlin_client.submit(
                    f"g.V().hasLabel('{lab}').limit({BATCH_SIZE}).id()"
                ).all().result()
                if not ids:
                    break
                id_str = ",".join(f"'{i}'" for i in ids)
                _gremlin_client.submit(f"g.V({id_str}).drop()").all().result()
                logger.info(f"  └ deleted {len(ids)} {lab} vertices")

    #----------------------------------------------------------------------------
    # AI Searchからドキュメントを削除（追加：ロールバック用）
    #----------------------------------------------------------------------------
    def delete_from_ai_search(self, doc_ids: List[str]) -> None:
        """AI Searchから指定されたドキュメントIDを削除"""
        if not doc_ids:
            return
            
        try:
            delete_docs = [{"id": doc_id} for doc_id in doc_ids]
            self.search_client.delete_documents(delete_docs)
            logger.info(f"⚠️AI Searchから{len(doc_ids)}件のドキュメントを削除しました")
        except Exception as e:
            logger.error(f"❌️AI Search削除エラー: {e}")

    #----------------------------------------------------------------------------
    # Gremlinからノードを削除（追加：ロールバック用）
    #----------------------------------------------------------------------------
    def delete_from_gremlin(self, gremlin_ids: List[str], _gremlin_client) -> None:
        """Gremlinから指定されたFileノードとその関連ノードを削除"""
        if not gremlin_ids:
            return
            
        for gremlin_id in gremlin_ids:
            try:
                # Fileノードとその関連ノード・エッジを削除
                _gremlin_client.submit(
                    f"g.V('{gremlin_id}').drop()"
                ).all().result()
                
                # 関連するMetaTypeとMetaInfoも削除
                for mt in self.meta_types:
                    mt_id = f"metatype_{gremlin_id}_{mt}"
                    _gremlin_client.submit(
                        f"g.V('{mt_id}').drop()"
                    ).all().result()

                logger.info("✅️Gemlin削除成功")
                    
            except Exception as e:
                logger.error(f"❌️Gremlin削除エラー {gremlin_id}: {e}")

    #----------------------------------------------------------------------------
    # バッチ単位でのトランザクション的な処理（追加）
    #----------------------------------------------------------------------------
    def process_batch_with_transaction(
        self, 
        batch_documents: List[Dict[str, Any]], 
        batch_idx: int,
        total_batches: int,
        target_customer_vectors: Dict[str, List[float]],
        business_challenge_vectors: Dict[str, List[float]]
    ) -> Dict[str, List[str]]:
        """
        バッチ単位でのトランザクション的な処理
        戻り値: {'ai_search': [成功したID], 'gremlin_tp': [成功したID], 'gremlin_prop': [成功したID]}
        """
        batch_success = {
            'ai_search': [],
            'gremlin_target_customer': [],
            'gremlin_business_challenge': []
        }
        
        logger.info(f"🔄 バッチ {batch_idx}/{total_batches} 処理開始（{len(batch_documents)}件）")
        
        # 1. AI Searchへの登録
        logger.info("📊 AI Search登録中...")
        for doc in batch_documents:
            try:
                upload_doc = doc.copy()
                upload_doc.pop('blob_url', None)
                self.search_client.upload_documents([upload_doc])
                batch_success['ai_search'].append(doc['id'])
                self.successful_documents['ai_search'].add(doc['id'])
                self.successful_documents['ai_search_gremlin_ids'].add(doc['gremlin_id'])
                logger.info(f"  ✅ AI Search登録成功: {doc['filename']}")
            except Exception as e:
                logger.error(f"  ❌ AI Search登録失敗 {doc['filename']}: {e}")
                self.failed_documents['ai_search'].append({
                    'document': doc,
                    'error': str(e)
                })
                
        # 2. AI Searchに成功したドキュメントのみGremlinに登録
        ai_search_success_ids = set(batch_success['ai_search'])
        gremlin_documents = [doc for doc in batch_documents if doc['id'] in ai_search_success_ids]
        
        if gremlin_documents:
            # 2.1 Gremlin Target Customer登録
            logger.info("📊 Gremlin Target Customer登録中...")
            try:
                # 該当ドキュメントのベクトルをフィルタリング
                filtered_tp_vectors = {
                    doc['gremlin_id']: target_customer_vectors[doc['gremlin_id']]
                    for doc in gremlin_documents
                    if doc['gremlin_id'] in target_customer_vectors
                }
                
                # Gremlinにバッチ登録（初回フラグ管理）
                if not hasattr(self, '_gremlin_tp_initialized'):
                    self._first_batch = True
                    self._gremlin_tp_initialized = True
                    
                self._create_gremlin_batch_nodes(
                    gremlin_documents, 
                    filtered_tp_vectors,
                    self.gremlin_client_target_customer, 
                    "target_customer"
                )
                
                # 成功したドキュメントを記録
                for doc in gremlin_documents:
                    batch_success['gremlin_target_customer'].append(doc['gremlin_id'])
                    self.successful_documents['gremlin_target_customer'].add(doc['gremlin_id'])

                logger.info("✅️Gemlin登録成功\n")
                    
            except Exception as e:
                logger.error(f"  ❌ Gremlin Target Customer登録失敗: {e}")
                for doc in gremlin_documents:
                    self.failed_documents['gremlin_target_customer'].append({
                        'document': doc,
                        'error': str(e)
                    })
                    
            # 2.2 Gremlin Business Challenge登録
            logger.info("📊 Gremlin Business Challenge登録中...")
            try:
                # 該当ドキュメントのベクトルをフィルタリング
                filtered_prop_vectors = {
                    doc['gremlin_id']: business_challenge_vectors[doc['gremlin_id']]
                    for doc in gremlin_documents
                    if doc['gremlin_id'] in business_challenge_vectors
                }
                
                # Gremlinにバッチ登録（初回フラグ管理）
                if not hasattr(self, '_gremlin_prop_initialized'):
                    self._first_batch = True
                    self._gremlin_prop_initialized = True
                    
                self._create_gremlin_batch_nodes(
                    gremlin_documents, 
                    filtered_prop_vectors,
                    self.gremlin_client_business_challenge, 
                    "business_challenge"
                )
                
                # 成功したドキュメントを記録
                for doc in gremlin_documents:
                    batch_success['gremlin_business_challenge'].append(doc['gremlin_id'])
                    self.successful_documents['gremlin_business_challenge'].add(doc['gremlin_id'])
                    
            except Exception as e:
                logger.error(f"  ❌ Gremlin Business Challenge登録失敗: {e}")
                for doc in gremlin_documents:
                    self.failed_documents['gremlin_business_challenge'].append({
                        'document': doc,
                        'error': str(e)
                    })
        
        # 3. 整合性チェックとロールバック処理
        # AI Searchには成功したがGremlinに失敗したドキュメントがある場合
        gremlin_failed_ids = []
        for doc in batch_documents:
            if (doc['id'] in batch_success['ai_search'] and 
                (doc['gremlin_id'] not in batch_success['gremlin_target_customer'] or
                 doc['gremlin_id'] not in batch_success['gremlin_business_challenge'])):
                gremlin_failed_ids.append(doc['id'])
                
        if gremlin_failed_ids:
            logger.warning(f"⚠️  整合性エラー検出: {len(gremlin_failed_ids)}件のドキュメントがAI Searchのみに存在")
            # オプション：AI Searchからロールバック
            self.delete_from_ai_search(gremlin_failed_ids)
            
        logger.info(f"✅ バッチ {batch_idx}/{total_batches} 処理完了")
        logger.info(f"   AI Search: {len(batch_success['ai_search'])}件成功")
        logger.info(f"   Gremlin TP: {len(batch_success['gremlin_target_customer'])}件成功")
        logger.info(f"   Gremlin Prop: {len(batch_success['gremlin_business_challenge'])}件成功")
        
        return batch_success

    #---------------------------------------------------------------------------------------------------------------
    # Gremlinにノードとエッジを作成（バッチ単位の処理用に修正）
    #---------------------------------------------------------------------------------------------------------------
    def _create_gremlin_batch_nodes(self, documents: List[Dict[str, Any]], tp_vectors: Dict[str, List[float]], _gremlin_client, mode: str) -> None:
        """バッチ単位でGremlinノードを作成（エラー時は例外をスロー）"""
        try:
            # 初回のみグラフクリア
            if hasattr(self, '_first_batch'):
                logger.info(f"Gremlinグラフの全頂点を削除中（{mode}）...")
                self._clear_graph_in_batches(_gremlin_client)
                delattr(self, '_first_batch')

            created_meta_info: set[str] = set()
            created_metatype:  set[str] = set()

            # ① Fileノード
            for doc in documents:
                _gremlin_client.submit(
                    f"""
                    g.addV('File')
                        .property('id','{doc["gremlin_id"]}')
                        .property('pk','file')
                        .property('filename','{doc["filename"].replace("'","''")}')
                        .property('filepath','{doc.get("filepath","")}')
                        .property('folder_type','{doc["folder_type"]}')
                        .property('ai_search_id','{doc["id"]}')
                        .property('blob_url','{doc.get("blob_url","")}')
                    """
                ).all().result()

            # ② FileごとにMetaTypeノード
            for doc in documents:
                file_id = doc["gremlin_id"]
                for mt in self.meta_types:
                    mt_id = f"metatype_{file_id}_{mt}"
                    mt_name = self.meta_type_mapping_reverse.get(mt, mt)
                    if mt_id not in created_metatype:
                        _gremlin_client.submit(
                            f"""
                            g.addV('MetaType')
                                .property('id','{mt_id}')
                                .property('pk','metatype')
                                .property('type','{mt}')
                                .property('name','{mt_name}')
                                .property('file_id','{file_id}')
                            """
                        ).all().result()
                        created_metatype.add(mt_id)

                    # File→has_metatype→MetaType
                    _gremlin_client.submit(
                        f"""
                        g.V('{file_id}')
                            .addE('has_metatype')
                            .to(g.V('{mt_id}'))
                        """
                    ).all().result()

            # ③ MetaTypeごとにMetaInfoノード
            for doc in documents:
                file_id = doc["gremlin_id"]
                for mt in self.meta_types:
                    mt_id = f"metatype_{file_id}_{mt}"
                    values = doc.get(mt, [])
                    values = values[:5]
                    for value in values:
                        if not value:
                            continue

                        # シングルクォーテーション対策
                        safe_value = json.dumps(value, ensure_ascii=False)

                        mi_id = f"metainfo_{mt_id}_{self._to_safe_id(value)}"
                        if mi_id not in created_meta_info:
                            _gremlin_client.submit(
                                f"""
                                g.addV('MetaInfo')
                                    .property('id','{mi_id}')
                                    .property('pk','metainfo')
                                    .property('meta_type','{mt}')
                                    .property('value',{safe_value})
                                    .property('file_id','{file_id}')
                                """
                            ).all().result()
                            created_meta_info.add(mi_id)

                        # MetaType→has_metainfo→MetaInfo
                        _gremlin_client.submit(
                            f"""
                            g.V('{mt_id}')
                                .addE('has_metainfo')
                                .to(g.V('{mi_id}'))
                            """
                        ).all().result()

        except Exception as e:
            logger.error(f"❌️ Gremlinバッチ登録エラー（{mode}）: {e}")
            raise

    #---------------------------------------------------------------------------------------------------------------
    # Gremlinに関連エッジを作成（全ドキュメント処理後に実行）
    #---------------------------------------------------------------------------------------------------------------
    def create_gremlin_edges(self, tp_vectors: Dict[str, List[float]], _gremlin_client, mode: str) -> None:
        """関連エッジを作成（全ドキュメント処理後）"""
        try:
            logger.info(f"🔍 create_gremlin_edges: mode={mode}, tp_vectors数={len(tp_vectors)}")
            success_set_key = 'gremlin_target_customer' if mode == "target_customer" else 'gremlin_business_challenge'
            logger.info(f"🔍 成功ドキュメント数（{success_set_key}）: {len(self.successful_documents[success_set_key])}")

            if 1 < len(tp_vectors):
                ids, vecs = zip(*tp_vectors.items())
                sims = cosine_similarity(vecs)
                edge_count = 0
                skipped_src = 0
                skipped_dst = 0
                skipped_sim = 0

                for i, src in enumerate(ids):
                    # 成功したドキュメントのみ処理
                    if src not in self.successful_documents[success_set_key]:
                        skipped_src += 1
                        logger.warning(f"  ⚠️ srcスキップ: {src} が successful_documents['{success_set_key}'] に含まれていません")
                        continue

                    top2 = np.argsort(sims[i])[::-1][1:3]
                    for j in top2:
                        if sims[i][j] <= 0:
                            skipped_sim += 1
                            continue
                        dst = ids[j]
                        # 相手も成功したドキュメントの場合のみエッジ作成
                        if dst not in self.successful_documents[success_set_key]:
                            skipped_dst += 1
                            logger.warning(f"  ⚠️ dstスキップ: {dst} が successful_documents['{success_set_key}'] に含まれていません")
                            continue

                        try:
                            src_mt = f"metatype_{src}_{mode}"
                            dst_mt = f"metatype_{dst}_{mode}"
                            logger.info(f"  ➕ エッジ作成: {src_mt} -[related_{mode}]-> {dst_mt} (sim={sims[i][j]:.4f})")
                            _gremlin_client.submit(
                                f"""
                                g.V('{src_mt}')
                                    .addE('related_{mode}')
                                    .to(g.V('{dst_mt}'))
                                    .property('similarity',{sims[i][j]:.4f})
                                """
                            ).all().result()

                            logger.info("✅️. エッジ作成成功")
                            edge_count += 1
                        except Exception as e:
                            logger.error(f"❌️ 関連エッジ作成エラー {src}->{dst}: {e}")
                                
                logger.info(f"✅ {edge_count}個の関連エッジを作成しました（{mode}） [skipped src={skipped_src}, dst={skipped_dst}, sim<=0={skipped_sim}]")
            else:
                logger.warning(f"⚠️ tp_vectorsが{len(tp_vectors)}件のためエッジ作成をスキップ（{mode}）")

        except Exception as e:
            logger.error(f"❌️ 関連エッジ処理エラー（{mode}）: {e}")
    
    #----------------------------------------------------------------------------
    # Azure Blobのneedsとseedsフォルダのファイル一覧取得
    #----------------------------------------------------------------------------
    def get_blob_files(self) -> List[Dict[str, str]]:
        files = []
        container_client = self.blob_client.get_container_client(self.container_name)
        
        for folder in ["needs", "seeds"]:
            blobs = container_client.list_blobs(name_starts_with=f"{folder}/")
            for blob in blobs:
                if not blob.name.endswith('/'):  # フォルダ除外
                    files.append({
                        "name": blob.name,
                        "path": blob.name,
                        "folder_type": folder
                    })
        
        return files

    #----------------------------------------------------------------------------
    # 処理結果のサマリーを出力（追加）
    #----------------------------------------------------------------------------
    def print_summary(self):
        """処理結果のサマリーを出力"""
        logger.info("\n" + "="*60)
        logger.info("📊 処理結果サマリー")
        logger.info("="*60)
        
        # 成功件数
        logger.info(f"✅ 成功件数:")
        logger.info(f"   AI Search: {len(self.successful_documents['ai_search'])}件")
        logger.info(f"   Gremlin Target Customer: {len(self.successful_documents['gremlin_target_customer'])}件")
        logger.info(f"   Gremlin Business Challenge: {len(self.successful_documents['gremlin_business_challenge'])}件")
        
        # 失敗件数
        logger.info(f"❌ 失敗件数:")
        logger.info(f"   AI Search: {len(self.failed_documents['ai_search'])}件")
        logger.info(f"   Gremlin Target Customer: {len(self.failed_documents['gremlin_target_customer'])}件")
        logger.info(f"   Gremlin Business Challenge: {len(self.failed_documents['gremlin_business_challenge'])}件")
        
        # 整合性チェック
        ai_only = (self.successful_documents['ai_search_gremlin_ids'] -
                   self.successful_documents['gremlin_target_customer'].intersection(
                       self.successful_documents['gremlin_business_challenge']))
        if ai_only:
            logger.warning(f"\n⚠️  整合性警告: {len(ai_only)}件のドキュメントがAI Searchのみに存在")
            logger.warning(f"   対象ID: {list(ai_only)[:5]}{'...' if len(ai_only) > 5 else ''}")
            
        # 失敗詳細（最初の5件）
        if any(self.failed_documents.values()):
            logger.info("\n📋 失敗詳細（最初の5件）:")
            for storage, failures in self.failed_documents.items():
                if failures:
                    logger.info(f"\n  {storage}:")
                    for failure in failures[:5]:
                        logger.info(f"    - {failure['document']['filename']}: {failure['error']}")
                        
        logger.info("="*60 + "\n")

    #-----------------------------------------------
    # 全ファイル処理のメイン処理（トランザクション処理対応）
    #-----------------------------------------------
    def process_all_files(self):
        logger.info("📝ファイル一覧取得中...")
        files = self.get_blob_files()
        logger.info(f"{len(files)} 個のファイルを処理します")
        
        documents = []
        target_customer_vectors = {}
        business_challenge_vectors = {}
        
        # ステップ1: 全ファイルの前処理（テキスト抽出、要約、メタ情報抽出）
        for i, file_info in enumerate(files):
            #-----------------------------------------------------------------
            # Debug(初回動作確認のため)登録確認出来たら下記2行をコメントアウト
            #-----------------------------------------------------------------
            # if 3 <= i:
            #   break
            #-----------------------------------------------------------------

            logger.info("="*80)
            logger.info(f"⌛️処理中 ({i+1}/{len(files)}): {file_info['name']}")
            
            # テキスト抽出
            content = self.extract_text_from_blob(file_info["path"])
            if not content:
                logger.warning(f"❌️テキスト抽出失敗: {file_info['name']}")
                logger.warning(f"右記のファイルはスキップして処理続行します: {file_info['name']}")
                continue
            logger.info("☑️ Blobファイルからのテキスト抽出完了")

            # 要約(content用)
            summary_content = self.summarize_by_gpt(content)
            logger.info("☑️ テキストの要約(content用)完了")

            # 要約(content_for_embed用)
            summary_content_for_embed = self.summarize_by_gpt_content_for_embed(content)
            logger.info("☑️ テキストの要約(content_for_embed用)完了")
            
            # Embedding取得
            embedding = self.get_embedding(summary_content_for_embed)
            logger.info("☑️ 要約のベクトル化完了")
            
            # メタ情報抽出
            meta_info = self.extract_meta_info(summary_content)
            logger.info("☑️ メタ情報抽出完了")
            
            # Blob SAS URL生成
            blob_url = self.generate_blob_sas_url(file_info["path"])
            logger.info("☑️ SASトークン付きURL生成完了")
            
            # ドキュメント作成
            doc_id = str(uuid.uuid4())
            gremlin_id = self.create_safe_id(file_info["path"])
            
            document = {
                "id": doc_id,
                "gremlin_id": gremlin_id,
                "filename": os.path.basename(file_info["name"]),
                "filepath": file_info["path"],
                "content": summary_content,
                "content_for_embed": summary_content_for_embed,
                "content_vector": embedding,
                "folder_type": file_info["folder_type"],
                "blob_url": blob_url,
                **meta_info
            }
            
            documents.append(document)
            
            # ベクトル生成
            if meta_info.get("target_customer"):
                target_customer_vector = self.create_target_metainfo_vector(meta_info["target_customer"])
                target_customer_vectors[gremlin_id] = target_customer_vector

            if meta_info.get("business_challenge"):
                business_challenge_vector = self.create_target_metainfo_vector(meta_info["business_challenge"])
                business_challenge_vectors[gremlin_id] = business_challenge_vector
        
        if not documents:
            logger.warning("❌️処理するファイルがありません")
            return
        
        # ステップ2: バッチ単位でのトランザクション登録処理
        logger.info("\n" + "="*60)
        logger.info("📤 データ登録フェーズ開始")
        logger.info("="*60)
        
        total_batches = (len(documents) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        
        for batch_idx in range(0, len(documents), self.BATCH_SIZE):
            batch_documents = documents[batch_idx:batch_idx + self.BATCH_SIZE]
            batch_num = (batch_idx // self.BATCH_SIZE) + 1
            
            # バッチ処理（トランザクション的）
            self.process_batch_with_transaction(
                batch_documents,
                batch_num,
                total_batches,
                target_customer_vectors,
                business_challenge_vectors
            )
            
            # APIレート制限対策
            if batch_idx + self.BATCH_SIZE < len(documents):
                logger.info(f"⏳ {self.SLEEP_TIME_BETWEEN_BATCHES}秒待機中...")
                time.sleep(self.SLEEP_TIME_BETWEEN_BATCHES)
        
        # ステップ3: 関連エッジの作成（全バッチ処理後）
        logger.info("\n" + "="*60)
        logger.info("🔗 関連エッジ作成フェーズ")
        logger.info("="*60)
        
        # Target Problemの関連エッジ作成
        logger.info("⌛️関連エッジ作成中（target_customer）...")
        self.create_gremlin_edges(
            target_customer_vectors,
            self.gremlin_client_target_customer,
            "target_customer"
        )
        
        # Propertyの関連エッジ作成
        logger.info("⌛️関連エッジ作成中（business_challenge）...")
        self.create_gremlin_edges(
            business_challenge_vectors,
            self.gremlin_client_business_challenge,
            "business_challenge"
        )
        
        # 処理結果のサマリー表示
        self.print_summary()
        
        logger.info("🎉 全体構築完了!")

#-----------------------------------
# エントリポイント
#-----------------------------------
if __name__ == "__main__":

    # 開始時刻
    start_time = time.time()

    # インスタンス生成
    batch = DataConstructionBatch()

    # 本処理
    batch.process_all_files()

    # 終了時刻
    elapsed_time = time.time() - start_time
    logger.info(f"⌚ 処理時間：{round(elapsed_time/60,0)}分")

