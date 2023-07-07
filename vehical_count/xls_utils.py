import os, xlwt
import xlsxwriter


font = xlwt.Font()
font.name = 'Times New Roman'
font.height = 20 * 14

alignment = xlwt.Alignment()
alignment.horz = 0x02
alignment.vert = 0x01

style_normal = xlwt.XFStyle()
style_normal.font = font
style_normal.alignment = alignment


def xls_save(path, num_list, car_num):
    current_time = os.path.basename(path)[:-10]

    workbook = xlwt.Workbook(encoding='utf-8')

    worksheet = workbook.add_sheet('sheet1')
    worksheet.col(0).width = 25 * 256
    for i in range(1, 4):
        worksheet.col(i).width = 20 * 256

    worksheet.write(0, 0, label='Time', style=style_normal)
    worksheet.write(0, 1, label='Camera 1', style=style_normal)
    worksheet.write(0, 2, label='Camera 2', style=style_normal)
    worksheet.write(0, 3, label='Camera merge', style=style_normal)

    worksheet.write(1, 0, label=current_time, style=style_normal)
    worksheet.write(1, 1, label=num_list[0], style=style_normal)
    worksheet.write(1, 2, label=num_list[1], style=style_normal)
    worksheet.write(1, 3, label=car_num, style=style_normal)

    workbook.save(os.path.join(os.path.dirname(path), current_time + '.xls'))

def xlsx_save(path, num_list):
    save_path = os.path.join(os.path.dirname(path), 'NumberofCars.xlsx')
    workbook  = xlsxwriter.Workbook(save_path)
    worksheet = workbook.add_worksheet()
    data = [num_list]
    worksheet.add_table('A1:C2', {'data': data, 'style': None,
                                  'columns': [{'header': 'Camera 1'},
                                              {'header': 'Camera 2'},
                                              {'header': 'Camera merge'}],})
    workbook.close()