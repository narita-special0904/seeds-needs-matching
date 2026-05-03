def render_target_customer():
    import streamlit as st
    import os
    import pandas as pd
    import numpy as np
    from typing import List, Dict, Any, Tuple
    import logging
    import traceback
    import json
    import tempfile

    # for Debug
    from pprint import pprint

    # ───────── Azure 関連 ─────────
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import (
        VectorizedQuery,
        QueryType,
        QueryCaptionType,
        QueryAnswerType,
    )
    from gremlin_python.driver import client, serializer
    from openai import AzureOpenAI

    # ───────── グラフ可視化 ─────────
    from pyvis.network import Network
    import streamlit.components.v1 as components

    # ───────── ロガー ─────────
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.WARNING
    )
    logger = logging.getLogger("【streamlit_ui】")

    st.markdown("📝**対象顧客観点機能**")

    #======================================================================
    # メインクラス
    #======================================================================
    class NeedsSeedsSearchUI:
        def __init__(self):

            # Azure AI Search
            self.search_client = SearchClient(
                endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                index_name=os.getenv("AZURE_AI_SEARCH_INDEX"),
                credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_KEY")),
            )

            # Cosmos DB for Apache Gremlin
            self.gremlin_client = client.Client(
                os.getenv("AZURE_COSMOSDB_GREMLIN_ENDPOINT"),
                "g",
                username=f"/dbs/{os.getenv('AZURE_COSMOSDB_GREMLIN_DB')}"
                f"/colls/{os.getenv('AZURE_COSMOSDB_GREMLIN_CONTAINER_TARGET_CUSTOMER')}",
                password=os.getenv("AZURE_COSMOSDB_GREMLIN_READ_WRITE_KEY"),
                message_serializer=serializer.GraphSONSerializersV2d0(),
            )

            # Azure OpenAI
            self.openai_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )

            # 起点ファイルノードの「対象顧客」メタタイプノードから、他の「対象顧客」メタタイプノードへ結合させる本数と、潜在ニーズ取得数
            self.n_limit = 2

        # ------------------------------------------------------------------
        # Embedding
        # ------------------------------------------------------------------
        def get_embedding(self, text: str) -> List[float]:
            try:
                rsp = self.openai_client.embeddings.create(
                    model=os.getenv("AZURE_OPENAI_MODEL_EMBEDDING_LARGE"),
                    input=text,
                )
                return rsp.data[0].embedding
            except Exception as e:
                logger.error(f"❌ Embedding error: {e}")
                return [0.0] * 3072

        # ------------------------------------------------------------------
        # Azure AI Search用クエリ圧縮
        # ------------------------------------------------------------------
        def _compress_query_for_ai_search(self, query: str) -> str:
            try:
                user_prompt = f"""### 指示 ###
                以下の質問文を、検索精度を高めるための短いクエリに圧縮してください（最大30文字程度）。	
                ただし以下の要件を必ず守ってください：
                1. 技術名、製品名、顧客名、用途、企業名などの**専門・固有表現は必ず残してください**。
                2. 「拡販」「展開」「活用」「探索」「獲得」などの**目的語や意図を示す動詞も省略しないでください**。
                3. 検索対象を明確にするため、**何をするための検索か**が伝わる表現を含めてください。
                4. 文法的に自然である必要はなく、検索語句として意味がつながっていれば構いません。

                ---
                質問文：{query}
                ---
                """

                res = self.openai_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                    messages=[
                        {"role": "system", "content": "あなたは高度な情報検索システムのクエリ最適化アシスタントです。"},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=200,
                    # temperature=0.2,
                )

                compressed_query = res.choices[0].message.content.strip()

                logger.info(f"🏷️ AI Search用圧縮クエリ：{compressed_query}")

                return compressed_query

            except Exception as e:
                logger.error(f"❌️Azure AI Search用クエリ圧縮エラー：{e}")

        # ------------------------------------------------------------------
        # Azure AI Search TOP-2
        # ------------------------------------------------------------------
        def search_documents(self, query: str) -> List[Dict]:
            # 調整項目
            K_NEAREST_NEIGHBORS = 8

            try:
                logger.info(f"質問：{query}")
                compressed_query = self._compress_query_for_ai_search(query)

                vec = self.get_embedding(query)
                vec_q = VectorizedQuery(vector=vec, k_nearest_neighbors=K_NEAREST_NEIGHBORS, fields="content_vector")

                results = self.search_client.search(
                    search_text=query,
                    vector_queries=[vec_q],
                    query_type=QueryType.SEMANTIC,
                    semantic_configuration_name="semantic-config",
                    select=[
                        "id",
                        "gremlin_id",
                        "filename",
                        "filepath",
                        "content",
                        "folder_type",
                        "company_name",
                        "location",
                        "business_segment",
                        "business_description",
                        "value_proposition",
                        "target_customer",
                        "market_industry",
                        "business_challenge",
                        "social_issue",
                        "technology_domain",
                        "rnd_theme",
                        "competitive_advantage",  
                    ],
                    # top=1,
                    top=2,
                )

                docs = []
                for r in results:
                    doc = {k: r.get(k, []) for k in r.keys()}
                    doc["score"] = r.get("@search.score", 0.0)
                    doc["reranker_score"] = r.get("@search.reranker_score", 0.0)
                    docs.append(doc)
                    logger.info(f"🏷️ {doc['filename']} (rerank {doc['reranker_score']})")

                return docs
            except Exception as e:
                logger.error(f"Search error: {e}")
                return []

        # ------------------------------------------------------------------
        # ファイルIDからメタ情報を取得する関数
        # ------------------------------------------------------------------
        def _get_file_meta_info(self, file_id: str) -> Dict[str, List[str]]:

            meta_info = {}

            meta_types = [
                "company_name", 
                "location", 
                "business_segment", 
                "business_description",
                "value_proposition",
                "target_customer",
                "market_industry",
                "business_challenge",
                "social_issue",
                "technology_domain",
                "rnd_theme",
                "competitive_advantage"
            ]

            try:
                for mt in meta_types:
                    # File → MetaType → MetaInfo の関係をたどってメタ情報を取得
                    # デバッグ: locationフィールドの内容を確認
                    # logger.info(f"📝_get_file_meta_info関数の引数のfile_id：{file_id} ")

                    query = f"""
                    g.V('{file_id}')
                    .out('has_metatype').has('type', '{mt}')
                    .out('has_metainfo')
                    .values('value')
                    .limit(5)
                    """
                    values = self.gremlin_client.submit(query).all().result()
                    if values:
                        meta_info[mt] = values
                    else:
                        meta_info[mt] = []
            except Exception as e:
                logger.error(f"Meta info fetch error for {file_id}: {e}")
                # エラー時は空のリストで初期化
                for mt in meta_types:
                    if mt not in meta_info:
                        meta_info[mt] = []
            
            return meta_info

        # ------------------------------------------------------------------
        # Gremlin traversal ＋ グラフ構築（メモリ上）
        # ------------------------------------------------------------------
        def traverse_related_target_customer(self, gremlin_id: str, origin_filename: str = "", is_primary: bool = True, other_origin_ids: List[str] = None) -> Tuple[List[Dict], Dict]:
            """起点ファイルから target_customer 関連をたどり、可視化グラフを組み立てる"""
            try:
                logger.info(f"🔄 traverse start: {gremlin_id}")

                # other_origin_idsが指定されていない場合は空リストに
                if other_origin_ids is None:
                    other_origin_ids = []

                graph = {
                    "origin_file_id": gremlin_id,
                    "nodes": [],
                    "edges": [],
                    "target_customer_nodes": {},  # File ID → target_customer graph node ID
                }

                # 起点 File ノード（プライマリかセカンダリかで色分け）
                origin_color = "#04063b" if is_primary else "#270480"

                graph["nodes"].append(
                    {
                        "id": gremlin_id,
                        "label": origin_filename or "起点ファイル",
                        "type": "origin" if is_primary else "origin_secondary",
                        "color": origin_color,  # 濃紺
                        "size": 30,
                    }
                )

                # 起点 File → target_customer → target_customer_nodes → File（limit N）
                simple_q = f"""
                g.V('{gremlin_id}')
                .out('has_metatype').has('type','target_customer')
                .outE('related_target_customer').as('e')
                .inV()
                .in('has_metatype').as('f')
                .where(__.as('f').id().is(neq('{gremlin_id}')))
                .select('e','f')
                .by('similarity')
                .by(valueMap(true))
                .limit({self.n_limit})
                """
                rs = self.gremlin_client.submit(simple_q).all().result()

                related_files: List[Dict] = []
                for idx, r in enumerate(rs):
                    sim = float(r.get("e", 0.0))
                    fd = r.get("f", {})
                    rid = fd.get("id") or fd.get("T.id")
                    if not rid:
                        continue

                    filename = self._get_val(fd, "filename", "")
                    blob_url = self._get_val(fd, "blob_url", "")

                    # メタ情報を取得
                    meta_info = self._get_file_meta_info(rid)

                    related_files.append(
                        {
                            "gremlin_id": rid,
                            "filename": filename,
                            "filepath": self._get_val(fd, "filepath", ""),
                            "folder_type": self._get_val(fd, "folder_type", ""),
                            "blob_url": blob_url,
                            "similarity": sim,
                            "meta_info": meta_info,  # 取得したメタ情報を設定
                        }
                    )

                    #-------------------------------------------------------------
                    # グラフに File ノード＋エッジ
                    #-------------------------------------------------------------
                    # 他の起点ノードでない場合のみ、関連ファイルとして追加
                    if rid not in other_origin_ids and not any(n["id"] == rid for n in graph["nodes"]):
                        graph["nodes"].append(
                            {
                                "id": rid,
                                "label": filename,
                                "type": "related",
                                "color": "#1f77b4",
                                "size": 25,
                            }
                        )


                    graph["edges"].append(
                        {
                            "from": gremlin_id,
                            "to": rid,
                            "label": f"類似度: {sim:.3f}",
                            "width": 5,
                            "color": "#ff0000",
                            "arrows": {"to": {"enabled": True, "scaleFactor": 1.2}},
                        }
                    )

                # 起点＋関連の File ID 一覧
                file_ids = [gremlin_id] + [f["gremlin_id"] for f in related_files]

                # 各 File 以下の MetaType/MetaInfo を追加（File ごとに重複保持）
                for fid in file_ids:
                    self._add_file_graph_structure(fid, graph)

                # 起点 target_customer → 他 File の target_customer へ強制2本エッジ
                self._link_target_customer_edges(graph)

                return related_files, graph

            except Exception as e:
                logger.error(f"❌️Traversal error: {e}")
                logger.error(traceback.format_exc())
                return [], {"nodes": [], "edges": []}

        # ------------------------------------------------------------------
        # File ごとの部分グラフを追加
        # ------------------------------------------------------------------
        def _add_file_graph_structure(self, file_id: str, graph: Dict, limit_metainfo: int = 5):
            """File → MetaType → MetaInfo を File 単位で重複登録して見やすく放射配置
            
            Args:
                file_id: ファイルID
                graph: グラフデータ
                limit_metainfo: 各MetaTypeごとに表示するMetaInfoの最大数（パフォーマンス改善）
            """
            # 除外するMetaType
            # excluded_types = ["location", "business_segment"]
            excluded_types = []
            
            try:
                mt_query = f"g.V('{file_id}').out('has_metatype').valueMap(true)"
                for mt in self.gremlin_client.submit(mt_query).all().result():
                    base_mt_id = self._get_val(mt, "id")
                    mt_type = self._get_val(mt, "type", "")
                    mt_name = self._get_val(mt, "name", "")

                    if not base_mt_id:
                        continue
                    
                    # 除外対象のMetaTypeはスキップ(2025/07/14)
                    if mt_type in excluded_types:
                        continue

                    # File 単位でユニーク化した MetaType ID
                    mt_id = f"{file_id}__{base_mt_id}"
                    
                    # MetaInfoの存在を事前にチェック(2025/07/14)
                    mi_check_query = f"g.V('{base_mt_id}').out('has_metainfo').limit(1)"
                    has_metainfo = self.gremlin_client.submit(mi_check_query).all().result()
                    
                    # MetaInfoが存在しない場合はMetaTypeノードも追加しない(2025/07/14)
                    if not has_metainfo:
                        continue

                    if not any(n["id"] == mt_id for n in graph["nodes"]):
                        graph["nodes"].append(
                            {
                                "id": mt_id,
                                "label": mt_name or "(metatype)",
                                "type": "metatype",
                                "color": "#a96bff" if mt_type == "target_customer" else "#2ca02c",
                                "size": 20,
                                "hidden": False,  # MetaTypeノードは常に表示
                            }
                        )
                        # target_customer ノードを控える
                        if mt_type == "target_customer":
                            graph["target_customer_nodes"][file_id] = mt_id

                    graph["edges"].append(
                        {
                            "from": file_id,
                            "to": mt_id,
                            "label": "has_metatype",
                            "color": "#888888",
                            "width": 2,
                        }
                    )

                    # -------- MetaInfo（制限付き）--------
                    # パフォーマンス改善: 各MetaTypeごとに最大limit_metainfo個まで表示
                    mi_query = f"g.V('{base_mt_id}').out('has_metainfo').valueMap(true).limit({limit_metainfo})"
                    for mi in self.gremlin_client.submit(mi_query).all().result():
                        base_mi_id = self._get_val(mi, "id")
                        mi_val = self._get_val(mi, "value", "")
                        if not base_mi_id or not mi_val:  # 値が空の場合はスキップ(2025/07/14)
                            continue

                        mi_id = f"{mt_id}__{base_mi_id}"

                        if not any(n["id"] == mi_id for n in graph["nodes"]):
                            graph["nodes"].append(
                                {
                                    "id": mi_id,
                                    "label": (mi_val[:20] + "...") if len(mi_val) > 20 else mi_val,
                                    "type": "metainfo",
                                    "color": "#d62728",
                                    "size": 12,
                                    "hidden": True,  # MetaInfoノードは初期状態で非表示(2025/07/14)
                                    "parent": mt_id,  # 親MetaTypeノードのIDを保持(2025/07/14)
                                }
                            )
                        graph["edges"].append(
                            {
                                "from": mt_id,
                                "to": mi_id,
                                "label": "has_metainfo",
                                "color": "#aaaaaa",
                                "width": 1,
                                "hidden": True,  # MetaInfoへのエッジも初期状態で非表示(2025/07/14)
                            }
                        )

            except Exception as e:
                logger.error(f"Graph-build error: {e}")

        # ----------------------------------------------------------------------------
        # target_customer ノード間に related_target_customer エッジN本を付与
        # ----------------------------------------------------------------------------
        def _link_target_customer_edges(self, graph: Dict):
            try:
                origin_fid = graph.get("origin_file_id")
                tp = graph.get("target_customer_nodes", {})
                if origin_fid not in tp:
                    return

                origin_tp_id = tp[origin_fid]
                # 起点以外の Nファイル分だけ
                others = [node_id for fid, node_id in tp.items() if fid != origin_fid][:self.n_limit]
                for dest in others:
                    if any(e["from"] == origin_tp_id and e["to"] == dest  for e in graph["edges"]):
                        continue
                    graph["edges"].append(
                        {
                            "from": origin_tp_id,
                            "to": dest,
                            "label": "related_target_customer",
                            "color": "#0066ff",
                            "width": 4,
                            "arrows": {"to": {"enabled": True}},
                        }
                    )
            except Exception as e:
                logger.error(f"Edge-link error: {e}")

        # ------------------------------------------------------------------
        # 「target_customer」MetaType間のエッジ追加
        # ------------------------------------------------------------------
        def _add_target_customer_edges(self, graph_data: Dict):
            try:
                mt_ids = {n["id"] for n in graph_data["nodes"] if n.get("type") == "metatype"}
                for mt_id in mt_ids:
                    rel_query = f"""
                    g.V('{mt_id}')
                    .outE('related_target_customer').as('e')
                    .inV().valueMap(true).as('v')
                    .select('e','v')
                    """
                    for r in self.gremlin_client.submit(rel_query).all().result():
                        dest = r.get("v", {})
                        dest_id = self._get_val(dest, "id")
                        if dest_id in mt_ids and not any(
                            e["from"] == mt_id and e["to"] == dest_id and e["label"] == "related_target_customer"
                            for e in graph_data["edges"]
                        ):
                            sim = self._get_val(r.get("e", {}), "similarity", 0.0)
                            graph_data["edges"].append(
                                {
                                    "from": mt_id,
                                    "to": dest_id,
                                    "label": "related_target_customer",
                                    "color": "#1f77b4",
                                    "width": 2,
                                    "title": f"similarity: {sim:.3f}",
                                }
                            )
            except Exception as e:
                logger.error(f"related_target_customer Edge add error: {e}")

        # ------------------------------------------------------------------
        # GraphSON → Python 値ユーティリティ
        # ------------------------------------------------------------------
        def _get_val(self, d: dict, key: str, default=None):
            if d is None:
                return default
            v = d.get(key, default)
            if isinstance(v, list):
                return v[0] if v else default
            return v

        # ------------------------------------------------------------------
        # スコア正規化
        # ------------------------------------------------------------------
        def normalize_score(self, score: float, score_type: str = "search") -> float:
            if score_type == "search":
                if score <= 0:
                    normalized = 0.10
                else:
                    normalized = 1 / (1 + np.exp(-0.6 * (score - 2.0)))
            else:  # similarity
                normalized = 1 / (1 + np.exp(-0.6 * (score - 2.0)))
            return round(normalized, 3)

        # ------------------------------------------------------------------
        # GPTモデルによる提案概要
        # ------------------------------------------------------------------
        def generate_proposal_summary(self, file_info: Dict, query: str) -> str:
            try:
                # meta_infoが存在する場合（潜在ニーズ）と存在しない場合（顕在ニーズ）の両方に対応
                meta = file_info.get("meta_info", {})
                
                # メタ情報の取得（meta_infoがある場合はそちらを優先）
                company_name          = meta.get('company_name', file_info.get('company_name', []))
                location              = meta.get('location', file_info.get('location', []))
                business_segment      = meta.get('business_segment', file_info.get('business_segment', []))
                business_description  = meta.get('business_description', file_info.get('business_description', []))
                value_proposition     = meta.get('value_proposition', file_info.get('value_proposition', []))
                target_customer       = meta.get('target_customer', file_info.get('target_customer', []))
                market_industry       = meta.get('market_industry', file_info.get('market_industry', []))
                business_challenge    = meta.get('business_challenge', file_info.get('business_challenge', []))
                social_issue          = meta.get('social_issue', file_info.get('social_issue', []))
                technology_domain     = meta.get('technology_domain', file_info.get('technology_domain', []))
                rnd_theme             = meta.get('rnd_theme', file_info.get('rnd_theme', []))
                competitive_advantage = meta.get('competitive_advantage', file_info.get('competitive_advantage', []))
                
                content = f"""
                質問: {query}

                ファイル情報:
                - ファイル名: {file_info.get('filename', '')}
                - 企業名: {', '.join(company_name)}
                - 所在地: {', '.join(location)}
                - 事業セグメント: {', '.join(business_segment)}
                - 事業内容: {', '.join(business_description)}
                - 提供価値: {', '.join(value_proposition)}
                - 対象顧客: {', '.join(target_customer)}
                - 経営課題: {', '.join(business_challenge)}
                - 市場・業界: {', '.join(market_industry)}
                - 社会課題: {', '.join(social_issue)}
                - 技術領域: {', '.join(technology_domain)}
                - 研究開発テーマ: {', '.join(rnd_theme)}
                - 競争優位性: {', '.join(competitive_advantage)}
                """

                rsp = self.openai_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT"),
                    messages=[
                        {"role": "system", "content": "あなたは優秀なコンサルタントとして提案に関しては世界レベルです。"},
                        {"role": "user", "content": "以下の質問とファイル情報から、提案概要を体言止め(末尾に「。」なし)で80文字以内で重要点を簡潔に生成してください。"},
                        {"role": "user", "content": "特に「経営課題」に着目してください。"},
                        {"role": "user", "content": "内容が中国語など外国語の場合は、日本語に翻訳してください。"},
                        {"role": "user", "content": "\n制約条件：一般的な誰でも思いつく内容ではなく、情報に基づいた具体性のあるコンサル視点を入れた内容にしてください。\n"},
                        {"role": "user", "content": content},
                    ],
                    max_completion_tokens=300,
                    # temperature=0.3,
                )

                # Debug
                # logger.info(f"generate_proposal_summaryのcontent：\n{content}")

                return rsp.choices[0].message.content
            except Exception as e:
                logger.error(f"提案概要生成エラー: {e}")
                return "提案概要の生成に失敗しました"

        # ------------------------------------------------------------------
        # GPT 総合解析
        # ------------------------------------------------------------------
        def generate_comprehensive_analysis(self, query: str, search_results: List[Dict], related_files: List[Dict], table_data: List[Dict[str, Any]]) -> str:
 
            try:
                # 顕在＋潜在 の情報を整形
                all_results: List[Dict[str, Any]] = []

                for r in search_results:
                    all_results.append(
                        {
                            "type": "顕在ニーズ",
                            "filename": r["filename"],
                            "company_name": r.get("company_name", []),
                            "location": r.get("location", []),
                            "business_segment": r.get("business_segment", []),
                            "business_description": r.get("business_description", []),
                            "value_proposition": r.get("value_proposition", []),
                            "target_customer": r.get("target_customer", []),
                            "market_industry": r.get("market_industry", []),
                            "business_challenge": r.get("business_challenge", []),
                            "social_issue": r.get("social_issue", []),
                            "technology_domain": r.get("technology_domain", []),
                            "rnd_theme": r.get("rnd_theme", []),
                            "competitive_advantage": r.get("competitive_advantage", []),
                        }
                    )


                for rf in related_files:
                    meta = rf.get("meta_info", {})
                    all_results.append(
                        {
                            "type": "潜在ニーズ",
                            "filename": rf["filename"],
                            "company_name": meta.get("company_name", []),
                            "location": meta.get("location", []),
                            "business_segment": meta.get("business_segment", []),
                            "business_description": meta.get("business_description", []),
                            "value_proposition": meta.get("value_proposition", []),
                            "target_customer": meta.get("target_customer", []),
                            "market_industry": meta.get("market_industry", []),
                            "business_challenge": meta.get("business_challenge", []),
                            "social_issue": meta.get("social_issue", []),
                            "technology_domain": meta.get("technology_domain", []),
                            "rnd_theme": meta.get("rnd_theme", []),
                            "competitive_advantage": meta.get("competitive_advantage", []),
                        }
                    )
                    

                
                results_text = ""
                for i, res in enumerate(all_results, 1):
                    results_text += f"""
                    {i}. {res['type']} - {res['filename']}
                    企業名: {', '.join(res['company_name'])}
                    所在地: {', '.join(res['location'])}
                    事業セグメント: {', '.join(res['business_segment'])}
                    事業内容: {', '.join(res['business_description'])}
                    提供価値: {', '.join(res['value_proposition'])}
                    対象顧客: {', '.join(res['target_customer'])}
                    市場・業界: {', '.join(res['market_industry'])}
                    経営課題: {', '.join(res['business_challenge'])}
                    社会課題: {', '.join(res['social_issue'])}
                    技術領域: {', '.join(res['technology_domain'])}
                    研究開発テーマ: {', '.join(res['rnd_theme'])}
                    競争優位性: {', '.join(res['competitive_advantage'])}
                    """

                system_prompt = """トップコンサルタントとして、具体的かつ実現可能で、かつ、新規ニーズにつながる提案を作成することが重要です。
                """


                prompt = f"""### 指示 ###
                質問に記載された内容に対して、
                それらを取り巻く「経営課題」から「顧客課題」、さらに「技術課題」へと、それぞれの関連性を示唆してください。
                なお、あなたからの次アクションの提案はしないでください。

                質問：
                {query}

                検索結果:
                {results_text}

                技術展開可能性：
                {table_data}

                ### 出力形式 ###
                以下のようにMarkDown形式で出力してください

                ## ■総論
                ここに総論を記載。総論は質問に対する手厚い深堀りした内容でお願いします。
                ただし、各論で示す「展開領域」について、そこに至る背景や理由も含めて、総合的に示してください。
                文字数は800文字程度で、視覚的に見やすい構成でお願いします。

                ### ■各論
                ここに【直接課題】と【間接課題】に分けて、展開領域候補を複数挙げます。
                展開領域候補毎に、以下に示す項目を整理して示してください。


                ### ◆直接課題
                #### 【展開領域候補】
                #### 【背景・市場課題】
                #### 【顧客課題】
                #### 【用途・業界】
                #### 【具体的活用案・期待効果】
                #### 【技術課題】

                ### ◆間接課題
                #### 【展開領域候補】
                #### 【背景・市場課題】
                #### 【顧客課題】
                #### 【用途・業界】
                #### 【具体的活用案・期待効果】
                #### 【技術課題】
                """

                rsp = self.openai_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=4000,
                    # temperature=0.2,
                )
                return rsp.choices[0].message.content
            except Exception as e:
                logger.error(f"❌️総合解析エラー: {e}")
                return "総合解析の生成に失敗しました"

        # ------------------------------------------------------------------
        # グラフ可視化
        # ------------------------------------------------------------------
        def visualize_graph(self, graph_data: Dict) -> str | None:
            try:
                if not graph_data["nodes"]:
                    return None

                # ノード数が多い場合の警告
                node_count = len(graph_data["nodes"])
                if 100 < node_count:
                    logger.warning(f"Graph has {node_count} nodes - rendering may be slow")

                net = Network(
                    height="900px",
                    width="100%",
                    bgcolor="#ffffff",
                    font_color="black",
                    directed=True,
                )

                # ノードを追加（hiddenフラグを考慮）
                for n in graph_data["nodes"]:
                    net.add_node(
                        n["id"],
                        label=n["label"],
                        color=n["color"],
                        size=n.get("size", 20),
                        title=f"{n['label']} ({n['type']})",
                        shape="dot",
                        hidden=n.get("hidden", False),  # 初期表示状態
                    )

                # エッジを追加（hiddenフラグを考慮）
                for e in graph_data["edges"]:
                    net.add_edge(
                        e["from"],
                        e["to"],
                        width=e.get("width", 2),
                        color=e.get("color", "#888888"),
                        arrows=e.get("arrows", {"to": {"enabled": True}}),
                        title=e.get("label", ""),
                        hidden=e.get("hidden", False),  # 初期表示状態
                    )

                # カスタムJavaScriptコードを含むオプション設定
                net.set_options(
                    """
                    var options = {
                    "physics": {
                        "enabled": true,
                        "solver": "forceAtlas2Based",
                        "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 200,
                        "springConstant": 0.08,
                        "damping": 0.4,
                        "avoidOverlap": 0.5
                        },
                        "maxVelocity": 50,
                        "minVelocity": 0.1,
                        "stabilization": {
                        "enabled": true,
                        "iterations": 500,
                        "updateInterval": 25,
                        "onlyDynamicEdges": false,
                        "fit": true
                        }
                    },
                    "interaction": {
                        "hideEdgesOnDrag": true,
                        "hideNodesOnDrag": false,
                        "hover": true,
                        "navigationButtons": true,
                        "keyboard": {
                        "enabled": true
                        }
                    },
                    "nodes": {
                        "font": {
                        "size": 12
                        }
                    },
                    "edges": {
                        "smooth": {
                        "type": "continuous"
                        }
                    }
                    }
                    """
                )

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".html", mode="w", encoding="utf-8"
                ) as tmp:
                    net.save_graph(tmp.name)
                    tmp_path = tmp.name

                # HTMLを読み込んで、クリックイベントのJavaScriptを追加
                with open(tmp_path, encoding="utf-8") as f:
                    html = f.read()
                
                # クリックイベントとフィット機能を追加するJavaScript
                custom_js = """
                <script type="text/javascript">
                window.addEventListener("load", function() {
                    // 全体表示
                    try { network.fit(); } catch(e) {}
                    
                    // MetaInfoノードの表示状態を管理するMap
                    var metaInfoVisibility = new Map();
                    
                    // クリックイベントの設定
                    network.on("click", function(params) {
                        if (params.nodes.length > 0) {
                            var clickedNodeId = params.nodes[0];
                            var clickedNode = nodes.get(clickedNodeId);
                            
                            // MetaTypeノードがクリックされた場合
                            if (clickedNode && clickedNode.title && clickedNode.title.includes("metatype")) {
                                var isVisible = metaInfoVisibility.get(clickedNodeId) || false;
                                metaInfoVisibility.set(clickedNodeId, !isVisible);
                                
                                // 関連するMetaInfoノードとエッジの表示/非表示を切り替え
                                var nodesToUpdate = [];
                                var edgesToUpdate = [];
                                
                                nodes.forEach(function(node) {
                                    // MetaInfoノードで、クリックされたMetaTypeノードの子ノードの場合
                                    if (node.title && node.title.includes("metainfo") && 
                                        node.id.startsWith(clickedNodeId + "__")) {
                                        nodesToUpdate.push({
                                            id: node.id,
                                            hidden: isVisible
                                        });
                                    }
                                });
                                
                                edges.forEach(function(edge) {
                                    // クリックされたMetaTypeノードから出るエッジ
                                    if (edge.from === clickedNodeId && 
                                        edge.to.startsWith(clickedNodeId + "__")) {
                                        edgesToUpdate.push({
                                            id: edge.id,
                                            hidden: isVisible
                                        });
                                    }
                                });
                                
                                // ノードとエッジを更新
                                nodes.update(nodesToUpdate);
                                edges.update(edgesToUpdate);
                            }
                        }
                    });
                });
                </script>
                </body>
                """
                html = html.replace("</body>", custom_js)
                
                # 更新されたHTMLを保存
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(html)
                
                return tmp_path

            except Exception as e:
                logger.error(f"Visualize error: {e}")
                logger.error(traceback.format_exc())
                return None

    # ======================================================================
    # Helper関数
    # ======================================================================
    # ------------------------------------------------------------------
    # 総合解析結果PDFダウンロード関数
    # ------------------------------------------------------------------
    def make_pdf_jp(text: str, font_path: str = "ipaexg.ttf") -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics
        from io import BytesIO
        import re

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # 日本語フォント登録
        pdfmetrics.registerFont(TTFont("IPAexGothic", font_path))
        c.setFont("IPAexGothic", 12)

        lines = text.splitlines()
        x, y = 50, height - 50
        max_width = width - 100
        font_size = 12

        for line in text.splitlines():
            # 1. 見出し対応（#削除＆大きく）
            m = re.match(r"^(#{1,6}) (.+)", line)
            if m:
                level = len(m.group(1))
                line = m.group(2)
                font_size = 22 - (level * 2)
            else:
                font_size = 12
            # 2. 箇条書き（-や*→・）
            if re.match(r"^[-*] ", line):
                line = "・" + line[1:]
            # 3. *や**を消す
            line = line.replace('**', '').replace('*', '')

            c.setFont("IPAexGothic", font_size)
            # 折り返し処理（省略したければこのままでOK）
            while pdfmetrics.stringWidth(line, "IPAexGothic", font_size) > max_width:
                for i in range(len(line), 0, -1):
                    if pdfmetrics.stringWidth(line[:i], "IPAexGothic", font_size) <= max_width:
                        break
                c.drawString(x, y, line[:i])
                y -= font_size + 4
                line = line[i:]
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("IPAexGothic", font_size)
            c.drawString(x, y, line)
            y -= font_size + 4
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("IPAexGothic", font_size)
        c.save()
        buffer.seek(0)
        return buffer.read()


    # ======================================================================
    # Streamlit アプリ
    # ======================================================================
    def main():

        # セッション状態
        for k, v in [
            ("target_customer_search_results", []),
            ("target_customer_related_files", []),
            ("target_customer_related_files_1", []),
            ("target_customer_related_files_2", []),
            ("target_customer_graph_data", {"nodes": [], "edges": []}),
            ("target_customer_analysis_result", ""),
            ("target_customer_pdf_bytes", ""),
            ("target_customer_table_data", []),
            ("target_customer_table_data_1", []),
            ("target_customer_table_data_2", []),
        ]:
            if k not in st.session_state:
                st.session_state[k] = v

        # クラスインスタンス生成
        ui = NeedsSeedsSearchUI()

        # 質問入力欄（画面切り替え時の入力値保持）
        query_key = "query_key_target_customer"
        if query_key not in st.session_state:
            st.session_state[query_key] = ""

        query = st.text_area(
            "ご質問を入力後「解析開始」ボタンを押下してください。",
            height=100,
            placeholder="例：建設業界の経営課題を教えてください。新規顧客開拓を目的としています。",
            value=st.session_state.get(query_key, "")
        )

        st.session_state[query_key] = query

        #----------------------------
        # --- 処理開始 ---
        #----------------------------
        if st.button("🔍 解析開始", type="primary") and query:
            with st.spinner("処理中...", show_time=True):
                # AI Search結果
                st.session_state.target_customer_search_results = ui.search_documents(query)

                if st.session_state.target_customer_search_results:
                    top    = st.session_state.target_customer_search_results[0]
                    second = st.session_state.target_customer_search_results[1]
                    st.info(f"Matching TOP1： {top['filename']}")
                    st.info(f"Matching TOP2： {second['filename']}")

                    # Gremlinグラフをトラバーサルした結果をsession_stateに保存
                    # TOP1
                    (st.session_state.target_customer_related_files_1, st.session_state.target_customer_graph_data) = ui.traverse_related_target_customer(
                        top["gremlin_id"], 
                        top["filename"], 
                        is_primary=True,
                        other_origin_ids=[second["gremlin_id"]]  # TOP2のIDを渡す
                    )  # TOP2の間接ファイルノードで色の上書き防止

                    # TOP2
                    (related_file_2, graph_data_2) = ui.traverse_related_target_customer(
                        second["gremlin_id"], 
                        second["filename"], 
                        is_primary=False,
                        other_origin_ids=[top["gremlin_id"]]  # TOP1のIDを渡す
                    )
                    st.session_state.target_customer_related_files_2 = related_file_2

                    #-----------------------------------------------------------------
                    # TOP2の間接ファイルノードで色の上書き防止
                    #-----------------------------------------------------------------
                    st.session_state.target_customer_graph_data["nodes"] += graph_data_2["nodes"]
                    st.session_state.target_customer_graph_data["edges"] += graph_data_2["edges"]


                    if st.session_state.target_customer_related_files_1 or st.session_state.target_customer_related_files_2:
                        st.success(f"関連ファイルを{len(st.session_state.target_customer_related_files_1)+len(st.session_state.target_customer_related_files_2)}件発見しました")
                    else:
                        st.warning("関連ファイルが見つかりませんでした")

                    # --- table_data構築 ---
                    table_data_1: List[Dict[str, Any]] = []
                    table_data_2: List[Dict[str, Any]] = []

                    # 顕在ニーズ
                    for res in st.session_state.target_customer_search_results[:2]:
                        if "meta_info" not in res:
                            res["meta_info"] = {
                                "company_name": res.get("company_name", []),
                                "location": res.get("location", []),
                                "business_segment": res.get("business_segment", []),
                                "business_description": res.get("business_description", []),
                                "value_proposition": res.get("value_proposition", []),
                                "target_customer": res.get("target_customer", []),
                                "market_industry": res.get("market_industry", []),
                                "business_challenge": res.get("business_challenge", []),
                                "social_issue": res.get("social_issue", []),
                                "technology_domain": res.get("technology_domain", []),
                                "rnd_theme": res.get("rnd_theme", []),
                                "competitive_advantage": res.get("competitive_advantage", []),
                            }


                        if res["filename"] == top["filename"]:
                            summary = ui.generate_proposal_summary(res, query)
                            table_data_1.append(
                                {
                                    "課題種別": "直接課題",
                                    "展開企業": ", ".join(res.get("company_name", [])[:6]),
                                    "技術領域": ", ".join(res.get("technology_domain", [])[:6]),
                                    "事業内容": ", ".join(res.get("business_description", [])[:6]),
                                    "対象顧客": ", ".join(res.get("target_customer", [])[:6]),
                                    "提供価値": ", ".join(res.get("value_proposition", [])[:6]),
                                    "経営課題": ", ".join(res.get("business_challenge", [])[:6]),
                                    "提案概要": summary,
                                    "ファイル名": res["filename"],
                                    "マッチングスコア": ui.normalize_score(res["reranker_score"], "search"),
                                    "URL": "",
                                }
                            )
                        else:
                            summary = ui.generate_proposal_summary(res, query)
                            table_data_2.append(
                                {
                                    "課題種別": "直接課題",
                                    "展開企業": ", ".join(res.get("company_name", [])[:6]),
                                    "技術領域": ", ".join(res.get("technology_domain", [])[:6]),
                                    "事業内容": ", ".join(res.get("business_description", [])[:6]),
                                    "対象顧客": ", ".join(res.get("target_customer", [])[:6]),
                                    "提供価値": ", ".join(res.get("value_proposition", [])[:6]),
                                    "経営課題": ", ".join(res.get("business_challenge", [])[:6]),
                                    "提案概要": summary,
                                    "ファイル名": res["filename"],
                                    "マッチングスコア": ui.normalize_score(res["reranker_score"], "search"),
                                    "URL": "",
                                }
                            )

                    # 潜在ニーズ
                    # TOP1の関連情報
                    for rel in st.session_state.target_customer_related_files_1:
                        summary = ui.generate_proposal_summary(rel, query)
                        table_data_1.append(
                            {
                                "課題種別": "間接課題",
                                "展開企業": ", ".join(rel.get("meta_info", {}).get("company_name", [])[:6]),
                                # "技術領域": ", ".join(rel.get("meta_info", {}).get("technology_domain", [])[:6]),
                                "事業内容": ", ".join(rel.get("meta_info", {}).get("business_description", [])[:6]),
                                "対象顧客": ", ".join(rel.get("meta_info", {}).get("target_customer", [])[:6]),
                                "提供価値": ", ".join(rel.get("meta_info", {}).get("value_proposition", [])[:6]),
                                "経営課題": ", ".join(rel.get("meta_info", {}).get("business_challenge", [])[:6]),
                                "提案概要": summary,
                                "ファイル名": rel["filename"],
                                "マッチングスコア": ui.normalize_score(rel["similarity"], "similarity"),
                                "URL": rel.get("blob_url", ""),
                            }
                        )

                    # TOP2の関連情報
                    for rel in st.session_state.target_customer_related_files_2:
                        summary = ui.generate_proposal_summary(rel, query)
                        table_data_2.append(
                            {
                                "課題種別": "間接課題",
                                "展開企業": ", ".join(rel.get("meta_info", {}).get("company_name", [])[:6]),
                                # "技術領域": ", ".join(rel.get("meta_info", {}).get("technology_domain", [])[:6]),
                                "事業内容": ", ".join(rel.get("meta_info", {}).get("business_description", [])[:6]),
                                "対象顧客": ", ".join(rel.get("meta_info", {}).get("target_customer", [])[:6]),
                                "提供価値": ", ".join(rel.get("meta_info", {}).get("value_proposition", [])[:6]),
                                "経営課題": ", ".join(rel.get("meta_info", {}).get("business_challenge", [])[:6]),
                                "提案概要": summary,
                                "ファイル名": rel["filename"],
                                "マッチングスコア": ui.normalize_score(rel["similarity"], "similarity"),
                                "URL": rel.get("blob_url", ""),
                            }
                        )

                    # 先頭ファイルの Blob URL 取得
                    if table_data_1:
                        try:
                            # TOP1
                            q = f"g.V('{st.session_state.target_customer_search_results[0]['gremlin_id']}').values('blob_url')"
                            r = ui.gremlin_client.submit(q).all().result()
                            if r:
                                table_data_1[0]["URL"] = r[0]

                        except Exception:
                            pass

                    if table_data_2:
                        try:
                            # TOP2
                            q = f"g.V('{st.session_state.target_customer_search_results[1]['gremlin_id']}').values('blob_url')"
                            r = ui.gremlin_client.submit(q).all().result()
                            if r:
                                table_data_2[0]["URL"] = r[0]

                        except Exception:
                            pass

                    # セッション保存
                    st.session_state.target_customer_table_data_1 = table_data_1
                    st.session_state.target_customer_table_data_2 = table_data_2
                    st.session_state.target_customer_table_data   = table_data_1 + table_data_2

                    st.session_state.target_customer_related_files = st.session_state.target_customer_related_files_1 + st.session_state.target_customer_related_files_2 


                    # --- 総合解析呼び出し（table_data含む） ---
                    with st.spinner("総合解析中...", show_time=True):
                        st.session_state.target_customer_analysis_result = ui.generate_comprehensive_analysis(
                            query,
                            st.session_state.target_customer_search_results,
                            st.session_state.target_customer_related_files,
                            st.session_state.target_customer_table_data,  # ←ここで画面の表内容を反映
                        )
                else:
                    st.error("検索結果が見つかりませんでした")

        # ───────── 提案候補テーブル ─────────
        if st.session_state.target_customer_search_results or st.session_state.target_customer_related_files:
            st.subheader("📋 技術展開可能性")
            st.caption("マッチングスコアは相対的なものです")
            table_data = st.session_state.get("target_customer_table_data", [])

            if table_data:
                df = pd.DataFrame(table_data)
                def make_link(row):
                    return (
                        f'<a href="{row.URL}" target="_blank">{row["ファイル名"]}</a>'
                        if row.URL
                        else row["ファイル名"]
                    )
                df["ファイル名"] = df.apply(make_link, axis=1)
                df = df.drop(columns=["URL"])

                #------------------------------------------------------------------------------------------------
                # CSSスタイルを追加して中央線（3行目と4行目の間）を太くする
                #------------------------------------------------------------------------------------------------
                st.markdown("""
                <style>
                .tech-table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 10px 0;
                }
                .tech-table th, .tech-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .tech-table th {
                    target_customer-color: #f2f2f2;
                    font-weight: bold;
                }
                /* 3行目（データ行の3番目）の下のボーダーを太くする */
                .tech-table tbody tr:nth-child(3) td {
                    border-bottom: 3px solid #39FF14 !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # HTMLテーブルを生成してクラス名を指定
                html_table = df.to_html(escape=False, index=False, classes="tech-table")
                st.markdown(html_table, unsafe_allow_html=True)
                #------------------------------------------------------------------------------------------------


                st.info(
                    f"検索結果: 直接課題 {sum(d['課題種別']=='直接課題' for d in table_data)} 件　"
                    f"間接課題 {sum(d['課題種別']=='間接課題' for d in table_data)} 件"
                )

        # ───────── タブ：総合解析 / ナレッジグラフ ─────────
        st.write("※最終結果出力まで5分程度お時間がかかる場合があります。")
        tab1, tab2 = st.tabs(["📊 総合解析結果", "🌐 ナレッジグラフ"])
        with tab1:
            if st.session_state.target_customer_analysis_result:
                st.markdown(st.session_state.target_customer_analysis_result)

                # PDFダウンロード機能
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                FONT_PATH = os.path.join(BASE_DIR, "fonts", "fonts-japanese-gothic.ttf")
                pdf_bytes = make_pdf_jp(st.session_state.target_customer_analysis_result, font_path=FONT_PATH)
                st.download_button(
                    label="総合解析結果PDFダウンロード",
                    data=pdf_bytes,
                    file_name="analysis_result.pdf",
                    mime="application/pdf",
                    icon="📝"
                )

        with tab2:
            if st.session_state.target_customer_graph_data["nodes"]:
                html = ui.visualize_graph(st.session_state.target_customer_graph_data)
                # Debug
                logger.info(f"HTMLサイズ：{round(os.path.getsize((html))/1024,1)} KB")

                if html:
                    with open(html, encoding="utf-8") as f:
                        components.html(f.read(), height=900, scrolling=True)
                    os.unlink(html)
            else:
                st.info("グラフデータがありません")

    #--------------------------------------
    # main関数実行
    #--------------------------------------
    main()