
# coding: utf-8

# In[18]:


import xlrd
import pymysql
from datetime import date, timedelta
# import file

data = xlrd.open_workbook('2018_Term Project_base data.xlsx')

# connect

db = pymysql.connect("localhost", "root", "10602141", "databaseproject",  charset='utf8')
cursor = db.cursor()

# loop insert

for k in range(0,7):
    table = data.sheets()[k]
    nrows = table.nrows
    ncols = table.ncols
    for i in range(1,nrows):
        #print (table.row_values(i))
        # book table 
        if k == 0 :  
            cursor.execute("INSERT INTO databaseproject.book( BookId, Title, PublisherName) VALUES( '%s', '%s', '%s');" % ( table.row_values(i)[0], table.row_values(i)[1], table.row_values(i)[2]))
            db.commit()

        # book copies table
        elif k == 1 :
            cursor.execute("INSERT INTO databaseproject.book_copies( BookId, BranchId, No_Of_Copies) VALUES( '%s', '%s', %d);" % ( table.row_values(i)[0], table.row_values(i)[1], table.row_values(i)[2]))
            db.commit()
        
        # book loan table
        elif k == 2 :
            try:
                outdays = table.row_values(i)[3]
                duedays = table.row_values(i)[4]
                start = date(1900,1,1)
                delta1 = timedelta(outdays-2)
                delta2 = timedelta(duedays-2)
                offset1 = start + delta1 
                offset2 = start + delta2
                cursor.execute("INSERT INTO databaseproject.book_loans( BookId, BranchId, CardNo, DateOut, DueDate) VALUES( '%s', '%s', '%s', %d, %d);" % ( table.row_values(i)[0], table.row_values(i)[1], table.row_values(i)[2], int(offset1.strftime('%Y%m%d')), int(offset2.strftime('%Y%m%d')) ))
                db.commit()
            
            except:
                cursor.execute("INSERT INTO databaseproject.book_loans( BookId, BranchId, CardNo, DateOut, DueDate) VALUES( '%s', '%s', '%s', NULL, NULL);" % ( table.row_values(i)[0], table.row_values(i)[1], table.row_values(i)[2] ))
                db.commit()
            
        # borrower table
        elif k == 3 :
            cursor.execute("INSERT INTO databaseproject.borrower( CardNo, Name, Address, Phone) VALUES( '%s', '%s', '%s', '%s');" % ( table.row_values(i)[0], table.row_values(i)[1], table.row_values(i)[2], table.row_values(i)[3]))
            db.commit()
        
        # library_branch talbe
        elif k == 4 :
            cursor.execute("INSERT INTO databaseproject.library_branch( BranchId, BranchName, Address) VALUES( '%s', '%s', '%s');" % ( table.row_values(i)[0], table.row_values(i)[1], table.row_values(i)[2]))
            db.commit()
            
        # book author table
        elif k == 5 :
            cursor.execute("INSERT INTO databaseproject.book_authors( BookId, AuthorName) VALUES( '%s', '%s');" % ( table.row_values(i)[0], table.row_values(i)[1]))
            db.commit()
            
        # publisher table
        elif k == 6 :
            cursor.execute("INSERT INTO databaseproject.publishernew( Name, Address, Phone) VALUES( '%s', '%s', '%s');" % ( table.row_values(i)[0], table.row_values(i)[1], table.row_values(i)[2] ))
            db.commit()

db.close()

