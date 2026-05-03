import os
import logging
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SearchSuggester,
    LexicalAnalyzerName,
    CorsOptions
)
from azure.core.credentials import AzureKeyCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("【index_create】")

from dotenv import load_dotenv
load_dotenv(".env", override=True)

class SearchIndexCreator:
    #-----------------------------------------
    # Azure AI Search クライアント初期化
    #-----------------------------------------
    def __init__(self):
        self.endpoint   = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.key        = os.getenv("AZURE_AI_SEARCH_ADMIN_KEY")
        self.index_name = os.getenv("AZURE_AI_SEARCH_INDEX")
    
        self.search_index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
        self.index_name = self.index_name
    
    #-----------------------------------------
    # インデックス構造の作成
    #-----------------------------------------
    def create_index(self):
        
        # インデックス定義(filenameをsortable=Trueにする理由 = メタ情報手動抽出時にfilenameのソート順に出力させるため)
        index = SearchIndex(
            name=self.index_name,
            fields=[
                # 基本フィールド
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="gremlin_id", type=SearchFieldDataType.String),
                SimpleField(name="folder_type", type=SearchFieldDataType.String),
                SearchableField(
                    name="filename",
                    type=SearchFieldDataType.String,
                    sortable=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT
                ),
                SimpleField(
                    name="filepath",
                    type=SearchFieldDataType.String,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT
                ),
                # GPT-5のコンテキスト用(800文字程度)
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT
                ),
                # ベクトル化対象の要約フィールド(600文字程度)
                SearchableField(
                    name="content_for_embed",
                    type=SearchFieldDataType.String,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT
                ),
                # メタタイプフィールド
                SearchableField(
                    name="company_name",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="location",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="business_segment",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="business_description",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="value_proposition",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="target_customer",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="market_industry",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="business_challenge",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="social_issue",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="technology_domain",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="rnd_theme",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                SearchableField(
                    name="competitive_advantage",
                    type=SearchFieldDataType.String, collection=True,
                    analyzer_name=LexicalAnalyzerName.JA_MICROSOFT,
                    facetable=True
                ),
                # ベクトルフィールド
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="vector-profile"
                )
            ],

            # CORS設定（全てのオリジンを許可する場合）
            cors_options = CorsOptions(
                allowed_origins=["*"],  # 全てのオリジンを許可
                max_age_in_seconds=300  # プリフライトリクエストのキャッシュ時間（オプション）
            ),
            
            # ベクトル検索設定
            vector_search=VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="hnsw-config"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-config",
                        kind="hnsw",
                        parameters={
                            "m": 10,
                            "efConstruction": 500,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ]
            ),           
            # セマンティック構成
            semantic_search = SemanticSearch(configurations=[
                SemanticConfiguration(
                    name="semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="content_for_embed"),
                        content_fields=[
                            SemanticField(field_name="content"),
                        ],
                        keywords_fields=[
                            SemanticField(field_name="company_name"),
                            SemanticField(field_name="location"),
                            SemanticField(field_name="business_segment"),
                            SemanticField(field_name="business_description"),
                            SemanticField(field_name="value_proposition"),
                            SemanticField(field_name="target_customer"),
                            SemanticField(field_name="market_industry"),
                            SemanticField(field_name="business_challenge"),
                            SemanticField(field_name="social_issue"),
                            SemanticField(field_name="technology_domain"),
                            SemanticField(field_name="rnd_theme"),
                            SemanticField(field_name="competitive_advantage"),
                        ]
                    )
                )
            ])
        )
        
        #-----------------------------------------
        # インデックス作成または更新
        #-----------------------------------------
        try:
            self.search_index_client.create_or_update_index(index)
            logger.info(f"Index '{self.index_name}' created/updated successfully")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    #-----------------------------------------
    # インデックス削除（開発時のみ使用）
    #-----------------------------------------
    def delete_index(self):
        try:
            self.search_index_client.delete_index(self.index_name)
            logger.info(f"✅️Index '{self.index_name}' deleted successfully")
        except Exception as e:
            logger.warning(f"Error deleting index: {e}")

# 実行
if __name__ == "__main__":
    
    creator = SearchIndexCreator()
    
    try:
        # インデックス構造作成
        creator.delete_index()
        creator.create_index()
        
        logger.info("✅️インデックス作成完了")
    except Exception as e:
        logger.error("❌️インデックス作成失敗")
