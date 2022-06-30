''' Duarte Emanuel Ramos Meneses - 2019216949
    Inês Martins Marçal - 2019215917
    Patricia Beatriz Silva Costa - 2019213995
    Multimédia - TP1 - 2021/2022              '''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.fftpack as fft
import math

test = False

mat = [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]

q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
       [12, 12, 14, 19, 26, 58, 60, 55],
       [14, 13, 16, 24, 40, 57, 69, 56],
       [14, 17, 22, 29, 51, 87, 80, 62],
       [18, 22, 37, 56, 68, 109, 103, 77],
       [24, 35, 55, 64, 81, 104, 113, 92],
       [49, 64, 78, 87, 103, 121, 120, 101],
       [72, 92, 95, 98, 112, 100, 103, 99]])

q_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
          [18, 21, 26, 66, 99, 99, 99, 99],
          [24, 26, 56, 99, 99, 99, 99, 99],
          [47, 66, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99],
          [99, 99, 99, 99, 99, 99, 99, 99]])


def menu_inicial():
    option = 0
    while (option < 1 or option > 3):
        print("Que imagem deseja ler?\n1- barn_mountains.bmp\n2- logo.bmp\n3- peppers.bmp\n")
        option=input("Opção: ")
        if option.isnumeric():
            option = int(option)
            if (option < 1 or option > 3):
                print("Opção inválida. Escolha outra opção.\n")
                option = 0
        else: 
            print("Opção inválida. Escolha outra opção.\n")
            option = 0  
    img_name = ''
    if (option == 1):
        img_name = 'barn_mountains.bmp'
    elif (option == 2):
        img_name = 'logo.bmp'
    else:
        img_name = 'peppers.bmp'

    return img_name


def show_img(title, img):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.show()


def show_colormap(img, colormap, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img, colormap)
    plt.show()


def colormap(r, g, b, name):
    return clr.LinearSegmentedColormap.from_list(name, [(0,0,0), (r,g,b)], 256)


def separate_rgb(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return R, G, B


def join_rgb(R, G, B, type):
    joined = np.zeros([R.shape[0], R.shape[1], 3], dtype=type)
    joined[:, :, 0] = R[:, :]
    joined[:, :, 1] = G[:, :]
    joined[:, :, 2] = B[:, :]

    return joined


def padding(R, G, B):  
    lines = R.shape[0]
    final_lines = 0
    if (lines % 16 != 0):
        final_lines = lines + (16 - lines % 16)
    else:
        final_lines = lines

    columns = R.shape[1]
    final_columns = 0
    if (columns % 16 != 0):
        final_columns = columns + (16 - columns % 16)
    else:
        final_columns = columns
    
    img_padd = np.zeros([final_lines, final_columns, 3], dtype=R.dtype)

    img_padd[:R.shape[0], :R.shape[1], 0] = R[:, :]  #R
    img_padd[:G.shape[0], :G.shape[1], 1] = G[:, :]  #G
    img_padd[:B.shape[0], :B.shape[1], 2] = B[:, :]  #B

    a = R.shape[0]
    b = R.shape[1]

    r_plus = np.repeat([R[-1, :]], (final_lines-a), axis=0)
    R = np.vstack((R, r_plus))

    g_plus = np.repeat([G[-1, :]], (final_lines-a), axis=0)
    G = np.vstack((G, g_plus))

    b_plus = np.repeat([B[-1, :]], (final_lines-a), axis=0)
    B = np.vstack((B, b_plus))

    r_plus = np.repeat(R[:, -1:], (final_columns-b), axis=1)
    R = np.hstack((R, r_plus))

    g_plus = np.repeat(G[:, -1:], (final_columns-b), axis=1)
    G = np.hstack((G, g_plus))

    b_plus = np.repeat(B[:, -1:], (final_columns-b), axis=1)
    B = np.hstack((B, b_plus))


    if test:
        img_padd = join_rgb(R, G, B, R.dtype)
        show_img("Padding", list(img_padd))
        print("Dimensão da imagem após o padding: " + str(img_padd.shape))

    return R, G, B


def reverse_padding(R, G, B, lines, columns):
    R_unp = R[:lines, :columns]
    G_unp = G[:lines, :columns]
    B_unp = B[:lines, :columns]

    if test:
        print("Dimensão da imagem após a reversão do padding: " + str(R_unp.shape))

    return R_unp, G_unp, B_unp


def YCbCr(R_p, G_p, B_p):

    if test:
        print("Pixel [0][0] antes da conversão de RGB para YCbCr\n\tValor original do R: "+str(R_p[0][0])+"\n\tValor original do G: "+str(G_p[0][0])+"\n\tValor original do B: "+str(B_p[0][0])+'\n')

    Y = mat[0][0]*R_p + mat[0][1]*G_p + mat[0][2]*B_p 
    Cb = mat[1][0]*R_p + mat[1][1]*G_p + mat[1][2]*B_p + 128
    Cr = mat[2][0]*R_p + mat[2][1]*G_p + mat[2][2]*B_p + 128

    if test:
        cmGray = colormap(1, 1, 1, 'myGray')
        show_colormap(Y, cmGray, "Canal Y da imagem com colormap cinzento")
        show_colormap(Cb, cmGray, "Canal Cb da imagem com colormap cinzento")
        show_colormap(Cr, cmGray, "Canal Cr da imagem com colormap cinzento")

    return Y, Cb, Cr


def reverse_ycbcr(Y, Cb, Cr):
    mat_i = np.linalg.inv(mat)

    R = (mat_i[0][0] * Y) + (mat_i[0][1] * (Cb - 128)) + (mat_i[0][2] * (Cr - 128))
    G = (mat_i[1][0] * Y) + (mat_i[1][1] * (Cb - 128)) + (mat_i[1][2] * (Cr - 128))
    B = (mat_i[2][0] * Y) + (mat_i[2][1] * (Cb - 128)) + (mat_i[2][2] * (Cr - 128))

    R = R.round()
    R[R>255] = 255
    R[R<0] = 0
    R = R.astype(np.uint8)

    G = G.round()
    G[G>255] = 255
    G[G<0] = 0
    G = G.astype(np.uint8)

    B = B.round()
    B[B>255] = 255
    B[B<0] = 0
    B = B.astype(np.uint8)

    if test:
        print("Pixel [0][0] após a conversão de YCbCr para RGB\n\tValor do R após conversão: "+str(R[0][0])+"\n\tValor do G após conversão: "+str(G[0][0])+"\n\tValor do B após conversão: "+str(B[0][0])+'\n')

    return R, G, B


def downsampling(Y, Cb, Cr, prop):
    #print(list(Cb[:8, :8]))

    if test:
        print("Dimensão Y pre downsampling:", Y.shape)
        print("Dimensão Cb pre downsampling:", Cb.shape)
        print("Dimensão Cr pre downsampling:", Cr.shape)

    Y_d = Y
    Cb_d = Cb
    Cr_d = Cr

    if prop == 0:
        Cb_d = Cb[0:Cb.shape[0]:2, :]
        Cr_d = Cr[0:Cr.shape[0]:2, :]

    Cb_d = Cb_d[:, 0:Cb_d.shape[1]:2]
    Cr_d = Cr_d[:, 0:Cr_d.shape[1]:2]

    if test:
        print("Dimensão da imagem com downsampling 4:2:"+ str(prop)+ " :", Cb_d.shape)

        print("Dimensão Y pos downsampling:", Y_d.shape)
        print("Dimensão Cb pos downsampling:", Cb_d.shape)
        print("Dimensão Cr pos downsampling:", Cr_d.shape)

    #print(list(Cb_d[:8, :8]))

    if test:
        cmGray = colormap(1,1,1,'Gray')

        show_colormap(Y_d, cmGray, 'Canal Y com downsampling 4:2:2 visto com colormap cinzento')
        print("Dimensão do canal Y após downsampling", Y_d.shape)
        show_colormap(Cb_d, cmGray, 'Canal Cb com downsampling 4:2:2 visto com colormap cinzento')
        print("Dimensão do canal Cb após downsampling", Cb_d.shape)
        show_colormap(Cr_d, cmGray, 'Canal Cr com downsampling 4:2:2 visto com colormap cinzento')
        print("Dimensão do canal Cr após downsampling", Cr_d.shape)

    return Y_d, Cb_d, Cr_d


def upsampling(Y_d, Cb_d, Cr_d, prop):
    Y_u = Y_d

    if prop == 0:
        Cb_u = np.repeat(Cb_d, 2, axis=1)
        Cb_u = np.repeat(Cb_u, 2, axis=0)

        Cr_u = np.repeat(Cr_d, 2, axis=1)
        Cr_u = np.repeat(Cr_u, 2, axis=0)

        #print(Cb_u)

    else:
        Cb_u = np.repeat(Cb_d, 2, axis=1)
        Cr_u = np.repeat(Cr_d, 2, axis=1)

    return Y_u, Cb_u, Cr_u


def dct(ch):
    dct = fft.dct(fft.dct(ch, norm="ortho").T, norm="ortho").T
    #dctLog = np.log(np.abs(dct) + 0.0001)
    #show_img("Canal com DCT", dctLog)

    return dct
    

def reverse_dct(ch):
    ch_rev = fft.idct(fft.idct(ch, norm="ortho").T, norm="ortho").T
    return ch_rev


def dct_blocks(Y, Cb, Cr, n_blocks):

    Y_dct = np.zeros((Y.shape[0], Y.shape[1]))
    Cb_dct = np.zeros((Cb.shape[0], Cb.shape[1]))
    Cr_dct = np.zeros((Cr.shape[0], Cr.shape[1]))

    for i in range(int(Y.shape[0]/n_blocks)):
        for j in range(int(Y.shape[1]/n_blocks)):
            Y_dct[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks] = dct(Y[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks])

    for i in range(int(Cb.shape[0]/n_blocks)):
        for j in range(int(Cb.shape[1]/n_blocks)):
            Cb_dct[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks] = dct(Cb[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks])
            Cr_dct[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks] = dct(Cr[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks])


    dctLogY = np.log(np.abs(Y_dct) + 0.0001)
    dctLogCb = np.log(np.abs(Cb_dct) + 0.0001)
    dctLogCr = np.log(np.abs(Cr_dct) + 0.0001)

    if test:
        cmGray = colormap(1, 1, 1, 'myGray')
        show_colormap(dctLogY, cmGray, "Canal Y com DCT " + str(n_blocks) + "x" + str(n_blocks) + " com colormap cinzento")
        show_colormap(dctLogCb, cmGray, "Canal Cb com DCT " + str(n_blocks) + "x" + str(n_blocks) + " com colormap cinzento")
        show_colormap(dctLogCr, cmGray, "Canal Cr com DCT " + str(n_blocks) + "x" + str(n_blocks) + " com colormap cinzento")

    return Y_dct, Cb_dct, Cr_dct


def q_matrix(quality):
    fac = 0

    if (quality < 50):
        fac = 50 / quality
    else:
        fac = (100-quality)/50
    
    if (fac != 0):
        q_yn = np.round(fac * q_y)
        q_cbcrn = np.round(fac * q_cbcr)
    else:
        q_yn = np.ones(q_y.shape)
        q_cbcrn = np.ones(q_y.shape)

    q_yn[q_yn>255] = 255
    q_yn[q_yn<1] = 1

    q_cbcrn[q_cbcrn>255] = 255
    q_cbcrn[q_cbcrn<1] = 1

    return q_yn, q_cbcrn


def reverse_dct_blocks(Y_dct, Cb_dct, Cr_dct, n_blocks):

    Y = np.zeros((Y_dct.shape[0], Y_dct.shape[1]))
    Cb = np.zeros((Cb_dct.shape[0], Cb_dct.shape[1]))
    Cr = np.zeros((Cr_dct.shape[0], Cr_dct.shape[1]))

    for i in range(int(Y_dct.shape[0]/n_blocks)):
        for j in range(int(Y_dct.shape[1]/n_blocks)):
            Y[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks] = reverse_dct(Y_dct[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks])

    for i in range(int(Cb_dct.shape[0]/n_blocks)):
        for j in range(int(Cb_dct.shape[1]/n_blocks)):
            Cb[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks] = reverse_dct(Cb_dct[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks])
            Cr[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks] = reverse_dct(Cr_dct[i*n_blocks:i*n_blocks+n_blocks, j*n_blocks:j*n_blocks+n_blocks])


    dctLogY = np.log(np.abs(Y) + 0.0001)
    dctLogCb = np.log(np.abs(Cb) + 0.0001)
    dctLogCr = np.log(np.abs(Cr) + 0.0001)

    if test:
        cmGray = colormap(1, 1, 1, 'myGray')
        show_colormap(dctLogY, cmGray, "Canal Y com DCT " + str(n_blocks) + "x" + str(n_blocks) + " revertido com colormap cinzento")
        show_colormap(dctLogCb, cmGray, "Canal Cb com DCT " + str(n_blocks) + "x" + str(n_blocks) + " revertido com colormap cinzento")
        show_colormap(dctLogCr, cmGray, "Canal Cr com DCT " + str(n_blocks) + "x" + str(n_blocks) + " revertido com colormap cinzento")

    return Y, Cb, Cr


def quant_dct(Y_dct, Cb_dct, Cr_dct,q_yn, q_cbcrn):

    for i in range(int(Y_dct.shape[0]/8)):
        for j in range(int(Y_dct.shape[1]/8)):
            Y_dct[i*8:i*8+8, j*8:j*8+8] = np.round(Y_dct[i*8:i*8+8, j*8:j*8+8]/q_yn)

    for i in range(int(Cb_dct.shape[0]/8)):
        for j in range(int(Cb_dct.shape[1]/8)):
            Cb_dct[i*8:i*8+8, j*8:j*8+8] = np.round(Cb_dct[i*8:i*8+8, j*8:j*8+8]/q_cbcrn)
            Cr_dct[i*8:i*8+8, j*8:j*8+8] = np.round(Cr_dct[i*8:i*8+8, j*8:j*8+8]/q_cbcrn)


    if test:
        quantLogY = np.log(np.abs(Y_dct) + 0.0001)
        quantLogCb = np.log(np.abs(Cb_dct) + 0.0001)
        quantLogCr = np.log(np.abs(Cr_dct) + 0.0001)

        
        cmGray = colormap(1, 1, 1, 'myGray')
        show_colormap(quantLogY, cmGray, "Canal Y com DCT quantizada 8x8 com colormap cinzento com qualidade 100")
        show_colormap(quantLogCb, cmGray, "Canal Cb com DCT quantizada 8x8 com colormap cinzento com qualidade 100")
        show_colormap(quantLogCr, cmGray, "Canal Cr com DCT quantizada 8x8 com colormap cinzento com qualidade 100")

    return Y_dct, Cb_dct, Cr_dct


def reverse_quant_dct(Y_dct, Cb_dct, Cr_dct, q_yn, q_cbcrn):

    for i in range(int(Y_dct.shape[0]/8)):
        for j in range(int(Y_dct.shape[1]/8)):
            Y_dct[i*8:i*8+8, j*8:j*8+8] = Y_dct[i*8:i*8+8, j*8:j*8+8] * q_yn

    for i in range(int(Cb_dct.shape[0]/8)):
        for j in range(int(Cb_dct.shape[1]/8)):
            Cb_dct[i*8:i*8+8, j*8:j*8+8] = Cb_dct[i*8:i*8+8, j*8:j*8+8] * q_cbcrn
            Cr_dct[i*8:i*8+8, j*8:j*8+8] = Cr_dct[i*8:i*8+8, j*8:j*8+8] * q_cbcrn

    return Y_dct, Cb_dct, Cr_dct


def dpcm(Y_qdct, Cb_qdct, Cr_qdct):
    Y_dpcm = np.copy(Y_qdct)
    Cb_dpcm = np.copy(Cb_qdct)
    Cr_dpcm = np.copy(Cr_qdct)

    for i in range(int(Y_qdct.shape[0]/8)):
        for j in range(int(Y_qdct.shape[1]/8)):
            if (i != 0):
                if (j != 0):
                    Y_dpcm[i*8, j*8] = Y_qdct[i*8, j*8] - Y_qdct[i*8, j*8-8]
                else:
                    Y_dpcm[i*8, j*8] = Y_qdct[i*8, j*8] - Y_qdct[i*8-8, int(Y_qdct.shape[1])-8]
            else:
                if (j != 0):
                    Y_dpcm[i*8, j*8] = Y_qdct[i*8, j*8] - Y_qdct[i*8, j*8-8]
            

    for i in range(int(Cb_qdct.shape[0]/8)):
        for j in range(int(Cb_qdct.shape[1]/8)):
            if (i!=0):
                if (j != 0):
                    Cb_dpcm[i*8, j*8] = Cb_qdct[i*8, j*8] - Cb_qdct[i*8, j*8-8]
                    Cr_dpcm[i*8, j*8] = Cr_qdct[i*8, j*8] - Cr_qdct[i*8, j*8-8]
                else:
                    Cb_dpcm[i*8, j*8] = Cb_qdct[i*8, j*8] - Cb_qdct[i*8-8, int(Cb_qdct.shape[1])-8]
                    Cr_dpcm[i*8, j*8] = Cr_qdct[i*8, j*8] - Cr_qdct[i*8-8, int(Cb_qdct.shape[1])-8]
            else:
                if (j != 0):
                    Cb_dpcm[i*8, j*8] = Cb_qdct[i*8, j*8] - Cb_qdct[i*8, j*8-8]
                    Cr_dpcm[i*8, j*8] = Cr_qdct[i*8, j*8] - Cr_qdct[i*8, j*8-8]


    if test:
        quantLogY = np.log(np.abs(Y_dpcm) + 0.0001)
        quantLogCb = np.log(np.abs(Cb_dpcm) + 0.0001)
        quantLogCr = np.log(np.abs(Cr_dpcm) + 0.0001)

        
        cmGray = colormap(1, 1, 1, 'myGray')
        show_colormap(quantLogY, cmGray, "Canal Y com DCT quantizada e DPCM 8x8 com colormap cinzento com qualidade 75")
        show_colormap(quantLogCb, cmGray, "Canal Cb com DCT quantizada e DPCM 8x8 com colormap cinzento com qualidade 75")
        show_colormap(quantLogCr, cmGray, "Canal Cr com DCT quantizada e DPCM 8x8 com colormap cinzento com qualidade 75")           

    return Y_dpcm, Cb_dpcm, Cr_dpcm


def reverse_dpcm(Y_dpcm, Cb_dpcm, Cr_dpcm):

    Y_qdct = np.copy(Y_dpcm)
    Cb_qdct = np.copy(Cb_dpcm)
    Cr_qdct = np.copy(Cr_dpcm)

    for i in range(int(Y_dpcm.shape[0]/8)):
        for j in range(int(Y_dpcm.shape[1]/8)):
            if (i!=0):
                if (j != 0):
                    Y_qdct[i*8, j*8] = Y_qdct[i*8, j*8-8] + Y_dpcm[i*8, j*8]
                else:
                    Y_qdct[i*8, j*8] = Y_qdct[i*8-8, int(Y_dpcm.shape[1])-8] + Y_dpcm[i*8, j*8]
            else:
                if (j != 0):
                    Y_qdct[i*8, j*8] = Y_qdct[i*8, j*8-8] + Y_dpcm[i*8, j*8]

    for i in range(int(Cb_dpcm.shape[0]/8)):
        for j in range(int(Cb_dpcm.shape[1]/8)):
            if (i!=0):
                if (j != 0):
                    Cb_qdct[i*8, j*8] = Cb_qdct[i*8, j*8-8] + Cb_dpcm[i*8, j*8] 
                    Cr_qdct[i*8, j*8] = Cr_qdct[i*8, j*8-8] + Cr_dpcm[i*8, j*8] 
                else:
                    Cb_qdct[i*8, j*8] = Cb_qdct[i*8-8, int(Cb_dpcm.shape[1])-8] + Cb_dpcm[i*8, j*8] 
                    Cr_qdct[i*8, j*8] = Cr_qdct[i*8-8, int(Cb_dpcm.shape[1])-8] + Cr_dpcm[i*8, j*8] 
            else:
                if (j != 0):
                    Cb_qdct[i*8, j*8] = Cb_qdct[i*8, j*8-8] + Cb_dpcm[i*8, j*8] 
                    Cr_qdct[i*8, j*8] = Cr_qdct[i*8, j*8-8] + Cr_dpcm[i*8, j*8] 

    return Y_qdct, Cb_qdct, Cr_qdct


def encoder(img_name, q_yn, q_cbcrn):
    plt.close('all')
    
    img = plt.imread(img_name)

    show_img("Imagem Original: " + img_name, img)

    if test:
        print("Dimensão original da imagem: " + str(img.shape))

    #Colormaps do RGB
    cmRed = colormap(1, 0, 0, 'myRed')
    cmGreen = colormap(0, 1, 0, 'myGreen')
    cmBlue = colormap(0, 0, 1, 'myBlue')

    if test:
        #Ver a imagem original com um colormap escolhido
        show_colormap(img, cmRed, "Imagem original com colormap " + "vermelho")

    #Separação dos canais RGB
    R, G, B = separate_rgb(img)

    if test:
        #Ver cada um dos canais com o colormap adequado
        show_colormap(R, cmRed, "Canal R da imagem original com colormap vermelho")
        show_colormap(G, cmGreen, "Canal G da imagem original com colormap verde")
        show_colormap(B, cmBlue, "Canal B da imagem original com colormap azul")

    #Padding da imagem
    R_p, G_p, B_p = padding(R, G, B)
    
    #Conversão para modelo YCbCr
    Y, Cb, Cr = YCbCr(R_p, G_p, B_p)

    Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr, 0)

    #Y_dct, Cb_dct, Cr_dct = dct(Y_d, Cb_d, Cr_d)

    Y_dctb, Cb_dctb, Cr_dctb = dct_blocks(Y_d, Cb_d, Cr_d, 8)

    Y_qdct, Cb_qdct, Cr_qdct = quant_dct(Y_dctb, Cb_dctb, Cr_dctb, q_yn, q_cbcrn)

    #print(Y_qdct[0:8, 8:16])

    Y_dpcm, Cb_dpcm, Cr_dpcm = dpcm(Y_qdct, Cb_qdct, Cr_qdct)

    #print(Y_dpcm[0:8, 8:16])
    
    return img, Y, Y_dpcm, Cb_dpcm, Cr_dpcm, img.shape[0], img.shape[1]


def decoder(Y, Cb, Cr, lines, columns, q_yn, q_cbcrn):

    Y_qdct, Cb_qdct, Cr_qdct = reverse_dpcm(Y, Cb, Cr)

    #print(Y_qdct[0:8, 8:16])

    Y_dct, Cb_dct, Cr_dct = reverse_quant_dct(Y_qdct, Cb_qdct, Cr_qdct, q_yn, q_cbcrn)

    Y_dctb, Cb_dctb, Cr_dctb = reverse_dct_blocks(Y_dct, Cb_dct, Cr_dct, 8)

    #Y_rdct, Cb_rdct, Cr_rdct = reverse_dct(Y, Cb, Cr)

    Y_u, Cb_u, Cr_u = upsampling(Y_dctb, Cb_dctb, Cr_dctb, 0)

    R, G, B = reverse_ycbcr(Y_u, Cb_u, Cr_u)
    if test:
        img_un_ycbcr = join_rgb(R, G, B, R.dtype)
        show_img("Conversão de YCbCr para RGB", list(img_un_ycbcr))

    R_unp, G_unp, B_unp = reverse_padding(R, G, B, lines, columns)
    if test:
        img_unp = join_rgb(R_unp, G_unp, B_unp, R_unp.dtype)
        show_img("Reversão do padding", img_unp)

    #Junção dos canais RGB de forma a obter a imagem original
    joined = join_rgb(R_unp, G_unp, B_unp, R_unp.dtype)
    show_img("Imagem descomprimida", list(joined))

    return Y_u, joined



def main():
    img_name = menu_inicial()
    q_yn, q_cbcrn = q_matrix(100)
    img_or, Y_or, Y, Cb, Cr, lines, columns = encoder(img_name, q_yn, q_cbcrn)
    Y_dec, decoded = decoder(Y, Cb, Cr, lines, columns, q_yn, q_cbcrn)

    img_or = img_or.astype(np.float64)

    cmGray = colormap(1, 1, 1, 'myGray')

    e = abs(Y_or - Y_dec)
    #print(e[0:8, 8:16])
    show_colormap(e, cmGray, "error")

    mse = (np.sum((img_or - decoded)**2)) / (lines*columns)
    print ("MSE:", mse)

    rmse = math.sqrt(mse)
    print ("RMSE:", rmse)

    pot = (np.sum((img_or)**2)) / (lines*columns)
    snr = 10 * math.log10(pot/mse)
    print ("SNR:", snr)

    psnr = 10 * math.log10((np.max(img_or)**2)/mse)
    print ("PSNR:", psnr)


if __name__ == "__main__":
    main()
