import pickle
from qdrant_client import QdrantClient, models
import json
import re

with open("prepare_database/brand_mapping.json", "r", encoding="utf-8") as f1:
    brand_mapping = json.load(f1)
brands = [brand.lower() for brand in brand_mapping.values()]

with open("prepare_database/category_mapping.json", "r", encoding="utf-8") as f2:
    category_mapping = json.load(f2)
categories = [category.lower() for category in category_mapping.values()]

def convert_to_number(value, unit):
    """Chuyển giá trị với đơn vị về số nguyên (đồng)"""
    number = float(value)
    if unit in ['triệu', 'm', 'củ']:
        return number * 1000000
    elif unit in ['nghìn', 'ngàn', 'k']:
        return number * 1000
    elif unit in ['đồng', 'd']:
        return number
    else:
        if number < 100:
            return number * 1000000 # Nếu không có đơn vị và nhỏ hơn 100, coi như là triệu
        else:
            return number * 1000 # Nếu không có đơn vị và lớn hơn 100, coi như là nghìn

def parse_query_to_filter(query):
    """Parse query tiếng việt để tạo filter cho Qdrant
    input: query (str),
    output: dict --> key: cột metadata (brand, category, price, stock, dictcount), values là dict range/match hoặc flag calculate"""

    filter = dict()
    query = query.lower()

    # 1. Kiểm tra Brand
    for brand in sorted(brands, key=len, reverse=True):
        if brand in query:
            filter['brand'] = brand
            break

    # 2. Kiểm tra Category
    for cat in sorted(categories, key=len, reverse=True):
        if cat in query:
            filter['category'] = cat
            break

    # Xử lý giá
    if 'giá rẻ' in query or 'gia re' in query or 'giá thấp' in query or 'giá tốt' in query or 'giá bình dân' in query:
        filter['price'] = {'gt': 0, 'lt': 3000000} # Dưới 3 triệu là rẻ
    elif 'giá cao' in query or 'giá đắt' in query or 'gia cao' in query or 'gia dat' in query:
        filter['price'] = {'gt': 30000000} # Trên 30 triệu là cao

    # Các pattern về giá
    price_patterns = [
        r'(?:từ|tu)\s*(\d+\.?\d*)\s*(triệu|trieu|nghìn|nghin|ngàn|ngan|đồng|dong|m|k|củ|cu|tr)?\s*(?:đến|den|tới|toi)\s*(\d+\.?\d*)\s*(triệu|trieu|nghìn|nghin|ngàn|ngan|đồng|dong|m|k|củ|cu|tr)?',
        r'(?:từ|tu)\s*(\d+\.?\d*)\s*(triệu|trieu|nghìn|nghin|ngàn|ngan|đồng|dong|m|k|củ|cu|tr)?',
        r'(?:trên|tren)\s*(\d+\.?\d*)\s*(triệu|trieu|nghìn|nghin|ngàn|ngan|đồng|dong|m|k|củ|cu|tr)?',
        r'(?:đến|den)\s*(\d+\.?\d*)\s*(triệu|trieu|nghìn|nghin|ngàn|ngan|đồng|dong|m|k|củ||tr)?',
        r'(?:dưới|duoi)\s*(\d+\.?\d*)\s*(triệu|trieu|nghìn|nghin|ngàn|ngan|đồng|dong|m|k|củ|cu|tr)?',
        r'(\d+\.?\d*)\s*(triệu|trieu|nghìn|nghin|ngàn|ngan|đồng|dong|m|k|củ|cu|tr)'
    ]

    # Tìm kiếm các pattern trong query va và tạo filter tương ứng
    for pattern in price_patterns:
        matches = re.findall(pattern, query)
        if matches:
            for match in matches:
                if 'đến' in pattern or 'den' in pattern or 'tới' in pattern or 'toi' in pattern:
                    if match[1] and match[2]:
                        min_price = convert_to_number(match[0], match[1])
                        max_price = convert_to_number(match[2], match[3])
                    elif match[1]:
                        min_price = convert_to_number(match[0], match[1])
                        max_price = convert_to_number(match[2], match[1])
                    elif match[3]:
                        min_price = convert_to_number(match[0], match[3])
                        max_price = convert_to_number(match[2], match[3])
                    else:
                        min_price = convert_to_number(match[0], 'triệu')
                        max_price = convert_to_number(match[2], 'triệu')
                    if min_price is not None and max_price is not None:
                        filter['price'] = {'gt': min_price, 'lt': max_price}
                elif ('từ' in pattern or 'tu' in pattern) and ('đến' not in pattern or 'den' not in pattern or 'tới' not in pattern or 'toi' not in pattern):
                    min_price = convert_to_number(match[0], match[1])
                    if min_price is not None:
                        filter['price'] = {'gt': min_price}
                elif ('đến' in pattern or 'den' in pattern or 'tới' in pattern or 'toi' in pattern) and ('từ' not in pattern or 'tu' not in pattern):
                    max_price = convert_to_number(match[0], match[1])
                    if max_price is not None:
                        filter['price'] = {'lt': max_price}
                elif 'dưới' in pattern or 'duoi' in pattern:
                    max_price = convert_to_number(match[0], match[1])
                    if max_price is not None:
                        filter['price'] = {'lt': max_price}
                elif 'trên' in pattern or 'tren' in pattern:
                    min_price = convert_to_number(match[0], match[1])
                    if min_price is not None:
                        filter['price'] = {'gt': min_price}
                else:
                    exact_price = convert_to_number(match[0], match[1])
                    if exact_price is not None:
                        filter['price'] = {'gte': exact_price, 'lte': exact_price}
            break

    # Xử lý stock còn hàng, hết hàng, hay còn X sản phẩm
    if 'còn hàng' in query or 'con hang' in query or 'có hàng' in query or 'co hang' in query or 'co san pham' in query or 'có sản phẩm' in query or 'còn sản phẩm' in query or 'còn không' in query or 'con khong' in query:
        filter['stock'] = {'gt': 0}
    elif 'hết hàng' in query or 'het hang' in query:
        filter['stock'] = {'eq': 0}

    stock_patterns = r"còn (\d+)\s*(sản phẩm|san pham|sp|cái|cai|chiếc|chiec)?"
    stock_matches = re.findall(stock_patterns, query)
    if stock_matches:
        stock_num = int(stock_matches[0][0])
        filter['stock'] = {'gte': stock_num}

    # Xử lý discount: giảm X%, giảm giá X tiền, giảm bao nhiêu
    disc_match = re.search(r'(giảm|giam|giảm giá| giam gia|khuyến mãi|khuyen mai)\s*(\d+(?:\.\d+)?)\s*%', query)
    # money_match = re.search(r'(giảm|giam|giảm giá| giam gia|khuyến mãi|khuyen mai)\s*(\d+(?:\.\d+)?)\s*(triệu|nghìn|ngàn|đồng|m|k|củ)?', query)
    if disc_match:
        disc_num = int(disc_match.group(2))
        filter['discount'] = disc_num
    # elif money_match:
    #     disc_money = convert_to_number(money_match.group(2)) if money_match.group(3) else float(money_match.group(2))
    #     filter['discount'] = disc_money
    if 'giảm bao nhiêu' in query or 'giam bao nhieu' in query or 'khuyến mãi bao nhiêu' in query or 'khuyen mai bao nhieu' in query:
        filter['calculate_discount'] = True

    # Xử lý màu sắc
    found_colors = []
    colors = ['đen', 'vàng', 'hồng', 'xanh lá', 'trắng', 'tím', 'xám', 'bạc', 'xanh dương', 'đỏ']
    for c in colors:
        if c in query:
            found_colors.append(c)

    if found_colors:
        # Nếu chỉ có 1 màu, lưu dạng string, nếu nhiều màu lưu dạng list
        filter['colors'] = found_colors if len(found_colors) > 1 else found_colors[0]

    return filter

class QdrantDB:
    def __init__(self, host="localhost", port=6333, api_key=None):
        self.host = host
        self.port = port
        self.client = QdrantClient(host=host, port=port, api_key=api_key)

    def init_collection(self, collection_name, vector_size, distance, drop=False, indexing=False):
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if drop and collection_name in names:
            self.client.delete_collection(collection_name)
            names.remove(collection_name)
            print("Đã xóa collection thành công")

        if collection_name not in names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(size=vector_size, distance=distance),
                },
                sparse_vectors_config={
                    "sparse_vector": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                }
            )

            if indexing:
                print(f"Tạo indexing cho collection {collection_name}")
                self._create_payload_indexes(collection_name)
                print("Đã tạo indexing cho collection", collection_name)

            print("Đã tạo collection thành công!")

    def _create_payload_indexes(self, collection_name):
        # Indexing for keyword fields
        keyword_fields = ['brand', 'category', 'type']
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        # Indexing numeric
        numeric_fields = ['price', 'discount', 'stock']
        for field in numeric_fields:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        # Indexing full name product
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name='name',
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.MULTILINGUAL,
                lowercase=True
            )
        )

        # Indexing list color
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="colors",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        print("Đã indexing toàn bộ các trường payload quan trọng")

    def _format_sparse_vector(self, sparse_dict):
        indices = [int(k) if str(k).isdigit() else k for k in sparse_dict.keys()]
        values = [float(v) for v in sparse_dict.values()]

        return models.SparseVector(indices=indices, values=values)

    def upload_data(self, collection_name, file_pkl, batch_size):
        with open(file_pkl, "rb") as f:
            data = pickle.load(f)

        points = []
        for item in data:
            point = models.PointStruct(
                id=item['id'],
                vector={
                    "dense_vector": item["dense_vector"],
                    "sparse_vector": self._format_sparse_vector(item["sparse_vector"])
                },
                payload=item["payload"],
            )
            points.append(point)

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

        print(f"Đã đẩy {len(points)} lên collection {collection_name}!")

    def hybrid_search(self, query, model, collection_name="products", k=50, search_filter=None):
        """
        Tạo Hybrid Search Filter cho Qdrant hỗ trợ Dense, Sparse và Metadata Filtering.
        """
        # Tạo Vector Query (Dense & Sparse) từ model BGE-M3
        dense_vec, sparse_dict = model.embedding(query)
        dense_vec = dense_vec.tolist()

        # Format Sparse Vector cho Qdrant
        sparse_obj = self._format_sparse_vector(sparse_dict)

        # Thực hiện Hybrid Search dùng Prefetch và Fusion (RRF)
        response = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense_vector",
                    limit=k,
                    filter=search_filter
                ),
                models.Prefetch(
                    query=sparse_obj,
                    using="sparse_vector",
                    limit=k,
                    filter=search_filter
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True
        )

        results = response.points
        return results

if __name__ == "__main__":
    # Khởi tạo client
    print("Khởi tạo QdrantDB")
    client = QdrantDB(host="localhost", port=6333)

    # Khởi tạo collection cho sản phẩm
    print("Khởi tạo collection")
    client.init_collection(collection_name="products", vector_size=1024, distance=models.Distance.COSINE, drop=True, indexing=True)
    # Khởi tạo collection cho policy
    client.init_collection(collection_name="policies", vector_size=1024, distance=models.Distance.COSINE, drop=True, indexing=False)

    # Upload data
    print("Upload data")
    client.upload_data(collection_name="products", file_pkl="prepare_database/chunks_product_bge_embedding.pkl", batch_size=64)
    client.upload_data(collection_name="policies", file_pkl="prepare_database/chunks_policy_embedding.pkl", batch_size=16)

    print("Hoàn thành")



