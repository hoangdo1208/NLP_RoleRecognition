# =================================================================
# PROJECT: Fine-tune PhoBert for Role Recognition
# PURPOSE: Fine-tune PhoBert model for role recognition tasks
# AUTHOR: K35 - Khoa Học Tích Hợp - Nhóm 2
# DATE: 2026
# =================================================================
import os
import pandas as pd
import json
import sqlite3

# =================================================================
# Define the NormalizeData class to handle data normalization tasks
# =================================================================
class NormalizeData:
    # data folder
    DATA_FOLDER: str = "./data"
    DATABASE_NAME: str = "NLPRoleRecognition"
    connection: sqlite3.Connection
    cursor: sqlite3.Cursor

    # =================================================================
    # Initialize the NormalizeData class
    # =================================================================
    def __init__(self):
        self.connection = sqlite3.connect(f"{self.DATA_FOLDER}/{self.DATABASE_NAME}.data")
        self.cursor = self.connection.cursor()

        # Initialize database schema
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS Utterance(
                conv_id INTEGER NOT NULL,
                turn_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                utterance TEXT NOT NULL,
                PRIMARY KEY(conv_id, turn_id)
            )
        """)

    # =================================================================
    # write data into database 
    # =================================================================
    def writeData(self, conv_id:int, turn_id:int, role:str, utterance:str):
        utterance_data = [conv_id, turn_id, role, utterance]
        self.cursor.execute("""
            INSERT OR IGNORE INTO Utterance(conv_id, turn_id, role, utterance)
            VALUES (?, ?, ?, ?)
        """, utterance_data)

    # =================================================================
    # Close SQLite database connection
    # =================================================================
    def close(self):
        self.connection.close()

    # =================================================================
    # Normalize the data by loading it from a Parquet file and processing the 'utterance' column
    # =================================================================
    def normalizeToCSV(self, data_file: str):
        # Load data from a Parquet file
        df = pd.read_parquet(data_file)

        with open(f"{self.DATA_FOLDER}/{data_file}.csv", "w", encoding="utf-8") as f:
            conv_id = 0
            f.writelines(f"conv_id, turn_id, role, utterance\n")
            for i in range(len(df['messages'])):
                conv_id = conv_id + 1           
                turn_id = 0
                for j in range(len(df['messages'][i])):
                    turn_id = turn_id + 1
                    utterance_json = df['messages'][i][j]
                    f.writelines(f"{conv_id}, {turn_id}, {utterance_json['role']}, {utterance_json['content']}\n")

    # =================================================================
    # Normalize the data by loading it from a Parquet file and write into database
    # =================================================================
    def normalize(self, data_file: str):
        # Load data from a Parquet file
        df = pd.read_parquet(data_file)

        # loop through all data to write into database
        conv_id = 0
        for i in range(len(df['messages'])):
            conv_id = conv_id + 1
            turn_id = 0
            for j in range(len(df['messages'][i])):
                turn_id = turn_id + 1
                utterance_json = df['messages'][i][j]
                self.writeData(conv_id, turn_id, utterance_json['role'], utterance_json['content'])
                print(f"Write data: {conv_id}, {turn_id}, {utterance_json['role']}, {utterance_json['content']} to datbase")

        # commit to database
        self.connection.commit()

    # =================================================================
    # Scan all file in the data folder and normalize to database
    # =================================================================
    def scanFileAndNormalize(self, data_folder: str):
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                full_path = os.path.join(root, file)
                print(f"Processing File: {file} | Full Path: {full_path}")
                self.normalize(full_path)

    # =================================================================
    # Show data from Database
    # =================================================================
    def showData(self):
        self.cursor.execute("SELECT * FROM utterance")
        rows = self.cursor.fetchall()
        for row in rows:
            #print(f"{row['conv_id']} | {row['turn_id']} | {row['role']} | {row['utterance']}")
            print(row)

# =================================================================
# Main execution block to run the NormalizeData class
# =================================================================
if __name__ == "__main__":
    # Example usage of the NormalizeData class
    #data_file = "../data/BlossomsAI Vietnamese Conversational Dataset/train-00025-of-00025.parquet"  # Specify the path to your Parquet file
    data_folder = "../data"  # Specify the path to your Parquet file
    normalizer = NormalizeData()
    normalizer.scanFileAndNormalize(data_folder)
    normalizer.close()