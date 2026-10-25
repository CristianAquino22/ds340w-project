Cristian Aquino - DS340W, Fall 2026

This project is a reproduction and extension of un-xPass for DS 340W. The code is based on ML-KULeuven/un-xPass by Robberechts, Van Roy & Davis (KDD 2023). This project uses the EURO 2020 dataset for training and testing, while using the World Cup 2022 data as the unseen set, split by match not by pass. I made this change because the paper used 2020/21 and 2021/22 Premier League data, which requires a paid StatsBomb subscription. Original code is Apache 2.0, see LICENSE.rst.

Environment:
Windows, Python 3.10 (3.13 breaks torch 1.13.1), multimethod < 2.0 and click < 8.2 pinned or the CLI crashes. 
