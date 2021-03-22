import pandas as pd
import numpy as np
from multiprocessing import Process
import sys
import math

rocket = 0
lisans_num_inputs = []
ad_soyad_inputs = []
elo_inputs = []
ukd_inputs = []
tur_sayisi_ = 0
bas_no = len(ad_soyad_inputs)
masa_no = math.ceil(bas_no/2)
puan = 0


def lisans_no():
    global rocket
    while rocket < sys.maxsize:
        while True:
            lisans_num = input("Oyuncunun lisans numarasini giriniz (bitirmek için 0 ya da negatif giriniz): ")
            if not lisans_num.isnumeric():
                print("Sadece sayı giriniz.")
                continue
            elif int(lisans_num) <= 0:
                baslangic_frame()
            else:
                lisans_num_inputs.append(lisans_num)
                break
        break
    rocket += 1
    return isim_soyisim()


def isim_soyisim():
    global rocket
    while rocket < sys.maxsize:
        while True:
            ad_soyad = input("Oyuncunun adini-soyadini giriniz:").upper()
            alphabet = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ "
            special_characters = "!@#$%^&*()-+?_=,<>/"

            if any(a in special_characters for a in ad_soyad):
                print("Özel karakterler kullanılamaz.")
                continue
            if not all(a in alphabet for a in ad_soyad):
                print("Sadece Türkçe harfler kullanınız.")
                continue
            else:
                ad_soyad_inputs.append(ad_soyad)
                break
        break
    rocket += 1
    return elo()


def elo():
    global rocket
    while rocket < sys.maxsize:
        while True:
            elo_ = input("Oyuncunun ELO’sunu giriniz (en az 1000, yoksa 0):")
            if int(elo_) < 1000:
                continue
            else:
                elo_inputs.append(elo_)
                break
        break
    rocket += 1
    return ukd()


def ukd():
    global rocket
    while rocket < sys.maxsize:
        while True:
            ukd_ = input("Oyuncunun UKD’sini giriniz (en az 1000, yoksa 0):")
            if int(ukd_) < 1000:
                continue
            else:
                ukd_inputs.append(ukd_)
                break
        break
    rocket += 1
    return lisans_no()


def baslangic_frame():
    baslangic_data = {
        'LNo': lisans_num_inputs,
        'Ad-Soyad': ad_soyad_inputs,
        'ELO': elo_inputs,
        'UKD': ukd_inputs
    }
    baslangic_df = pd.DataFrame(baslangic_data)
    baslangic_df['BSNo'] = np.arange(1, len(baslangic_df) + 1)
    baslangic_df.set_index('BSNo', inplace=True)
    baslangic_df_sorted = baslangic_df.sort_values(by='LNo', ascending=True)
    baslangic_df_sorted_ = baslangic_df_sorted.sort_values(by='Ad-Soyad')
    baslangic_df_sorted_f = baslangic_df_sorted_.sort_values(by='ELO', ascending=False)
    baslangic_df_sorted_final = baslangic_df_sorted_f.sort_values(by='UKD', ascending=False)
    print(baslangic_df_sorted_final)
    return tur_sayisi()


def tur_sayisi():
    global rocket
    while rocket < sys.maxsize:
        while True:
            tur = input("Turnuvadaki tur sayisini giriniz:")
            oyuncu_sayisi = math.log(len(ad_soyad_inputs), 2)
            yukari = math.ceil(oyuncu_sayisi)
            if int(tur) < yukari or int(tur) > (len(ad_soyad_inputs)-1):
                print("Tur sayısı bu olamaz:")
                continue
            else:
                tur = tur_sayisi_
                break
        break
    rocket += 1
    return renk()


def renk():
    global rocket
    while rocket < sys.maxsize:
        while True:
            baslangic_renk = input("Baslangic siralamasina gore ilk oyuncunun ilk turdaki rengini giriniz (b/s):")
            if baslangic_renk != 's':
                print("BSNo'su tek olanlar siyah olmalıdır.")
                continue
            else:
                tur_1_eslestirme_frame()
                break
        break
    rocket += 1


def tur_1_eslestirme_frame():
    tur1_data = {
        'LNo': lisans_num_inputs,
        '-': None,
        'Puan': 0.00,
    }
    tur1_df = pd.DataFrame(tur1_data)

    tur1_df['MNo'] = np.arange(1, len(tur1_df) + 1)
    tur1_df.set_index('MNo', inplace=True)
    tur1_df['BNo'] = bas_no
    beyaz_lno = bas_no % 2 == 0
    siyah_lno = bas_no % 2 == 1
    tur1_df.loc[:, 'Siyahlar LNo'] = siyah_lno
    tur1_df.loc[:, 'Beyazlar LNo'] = beyaz_lno
    print(tur1_df)
    return tur_1()


def tur_1():
    global rocket
    while rocket < sys.maxsize:
        while True:
            for i in range(0, masa_no):
                mac_sonucu = input('1. turda' + i + '. masada oynanan macin sonucunu giriniz (0-5):')
                if int(mac_sonucu) < 0 or int(mac_sonucu) > 5:
                    continue
                elif int(mac_sonucu) == 0:
                    print('Beraberlik')
                elif int(mac_sonucu) == 1:
                    print('Beyaz galip')
                elif int(mac_sonucu) == 2:
                    print('Siyah galip')
                elif int(mac_sonucu) == 3:
                    print('Siyah maça gelmemiş')
                elif int(mac_sonucu) == 4:
                    print('Beyaz maça gelmemiş')
                elif int(mac_sonucu) == 5:
                    print('Her iki oyuncu da maça gelmemiş')
            else:
                break
        break
    rocket += 1


if __name__ == '__main__':
    p1 = Process(target=lisans_no())
    p1.start()
    p2 = Process(target=isim_soyisim())
    p2.start()
    p3 = Process(target=elo())
    p3.start()
    p4 = Process(target=ukd())
    p4.start()
    p5 = Process(target=renk())
    p5.start()
    p6 = Process(target=tur_sayisi())
    p6.start()
    p7 = Process(target=tur_1())
    p7.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
