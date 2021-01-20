import mysql.connector
import pandas as pd


class Store:
    def __init__(self, store_name):
        self.name = store_name
        '''
        self.aws_host = ""
        self.aws_user = ""
        self.aws_password = ""
        self.aws_db = "grocerymatedb"
        self.db = mysql.connector.connect(self.aws_host="",
            self.user="", self.password="",
            self.aws_db="grocerymatedb")
        '''
        self.data = self.get_store_product_data()
        self.product_categories = self.data.drop_duplicates().category.tolist()
        self.products = self.data.drop_duplicates()['product'].tolist()
        

    def get_store_product_data(self, display=True):
        '''
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM shop_1")
        store_db = cursor.fetchall()
        '''
        store_db = [
            ['haagen dazs', 'ice cream', '21'],
            ['oreo', 'cookies', '12'],
            ['pudding', 'dessert', 'freezer'],
            ['chocolate milk', 'milk', 'freezer'],
            ['lays', 'snacks', '11'],
            ['red bull', 'drinks', '10'],
            ['toilet paper', 'paper', '2'],
            ['croissants', 'bread', 'bakery'],
            ['coco-cola', 'drinks', '10'],
            ['eggs', 'eggs', 'freezer']
            ]
        store_df = pd.DataFrame(store_db,
                    columns=['product', 'category', 'aisle'])
        if display:
            print(store_df, "\n")
        return store_df


    def extract_products_from_message(self, message, pattern=", "):
        products = message.split(pattern)
        #print(products)
        return products
        

    def locate_products(self, msg_str):
        products = self.extract_products_from_message(msg_str)
        product_dict = {}
        for product in products:
            if product in self.product_categories:
                aisle = self.data.loc[self.data['category']==product]['aisle'].tolist()[0]
                print(product + " could be found in aisle:", aisle)
                product_dict[product] = aisle
            elif product in self.products:
                aisle = self.data.loc[self.data['product']==product]['aisle'].tolist()[0]
                print(product + " could be found in aisle:", aisle)
                product_dict[product] = aisle
            else:
                print("could not find the aisle number for", product)
                product_dict[product] = "unknown"
        
        aisle_dict = {}
        for product in product_dict:
            product_aisle = product_dict[product]
            if product_aisle not in aisle_dict:
                aisle_dict[product_aisle] = [product]
            elif product_aisle in aisle_dict:
                aisle_dict[product_aisle].append(product)
        # print(aisle_dict)  
        return aisle_dict


