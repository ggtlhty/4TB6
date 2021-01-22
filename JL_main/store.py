import mysql.connector
import pandas as pd


class Store:
    def __init__(self, storeName):
        self.name = storeName
        '''
        self.aws_host = ""
        self.aws_user = ""
        self.aws_password = ""
        self.aws_db = "grocerymatedb"
        self.db = mysql.connector.connect(self.aws_host="",
            self.user="", self.password="",
            self.aws_db="grocerymatedb")
        '''
        self.data = self.getStoreProductData()
        self.productCategories = self.data.drop_duplicates().category.tolist()
        self.products = self.data.drop_duplicates()['product'].tolist()
        

    def getStoreProductData(self, display=True):
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


    def extractProductsFromMessage(self, message, pattern=", "):
        products = message.split(pattern)
        #print(products)
        return products
        

    def locateProducts(self, msg_str):
        products = self.extractProductsFromMessage(msg_str)
        productDict = {}
        for product in products:
            product = product.lower()
            if product in self.productCategories:
                aisle = self.data.loc[self.data['category']==product]['aisle'].tolist()[0]
                print(product + " could be found in aisle:", aisle)
                productDict[product] = aisle
            elif product in self.products:
                aisle = self.data.loc[self.data['product']==product]['aisle'].tolist()[0]
                print(product + " could be found in aisle:", aisle)
                productDict[product] = aisle
            else:
                print("could not find the aisle number for", product)
                productDict[product] = "unknown"
        
        aisleDict = {}
        for product in productDict:
            productAisle = productDict[product]
            if productAisle not in aisleDict:
                aisleDict[productAisle] = [product]
            elif productAisle in aisleDict:
                aisleDict[productAisle].append(product)
        # print(aisle_dict)  
        return aisleDict


