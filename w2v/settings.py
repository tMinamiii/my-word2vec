NUM_UNITS = 2000
DATA_FILE = './data/vector/tfidf/tfidf.data'
METADATA_FILE = './data/vector/tfidf/tfidf.meta'
LOG_FILE = '/tmp/yn_categories_logs'
BATCH_SIZE = 500
TOTAL_STEP = 1000000
LEARNING_RATE = 0.0001  # 学習率
TRAINING_DATA_RATIO = 0.8  # 全データのうち訓練用に使う割合
MINIMUM_MANUSCRIPT_LENGTH = 300
MINIMUM_TOKEN_LENGTH = 150
KEEP_PROB = 0.5
# smaller equal than BATCH_SIZE
# 最大にしたい場合はNoneを設定する
PCA_DIMENSION = 100
# CATEGORIES = ['IT総合', '映画', '経済総合', '野球',
#              '社会', 'ライフ総合', 'エンタメ総合', 'サッカー', 'スポーツ総合']

# CATEGORIES = ['映画', '経済総合', '野球',
#              '社会', 'ライフ総合', 'エンタメ総合', ]
CATEGORIES = ['IT総合']
PCA_BATCH_DATA_LENGTH = 150
FTP_SERVER = '172.16.27.200'
FTP_USER = 'crawlerpy'
FTP_PASS = 'crawlerpy'
FTP_NEWS_DIR = 'Crawler/YahooNews'
FTP_TOKEN_DIR = 'Crawler/TokenFiles'
