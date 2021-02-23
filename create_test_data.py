from pprint import pprint
import sys

COLUMN_NAMES = ['col1', 'col2', 'complexAge', 'totalRooms', 'totalBedrooms', 'complexInhabitants', 'apartmentsNr', 'col8', 'medianCompexValue']
DATASET_PATH = 'apartmentComplexData.txt'

if __name__=='__main__':
    if len(sys.argv)==2:
        try:
            line_nr = int(sys.argv[1])
        except:
            print("Error! the second argument should represent the line_nr (an integer)")
            quit()
    else:
        line_nr = 0
        
    print("The line nr of data to be analyzed:", line_nr)
    f = open(DATASET_PATH, 'r')


    line_of_data = f.readlines()[line_nr]

    data_splitted = line_of_data.replace("\n", '').split(",")
    print('line of data:', line_of_data)


    test_data = { "data":[ {} ] }

    for idx, col in enumerate(COLUMN_NAMES):
        if col=='medianCompexValue':
            real_value = data_splitted[idx]
        else:
            test_data["data"][0][col] = data_splitted[idx]

    print("test data:")
    pprint(test_data)
    print("real value:", real_value)

