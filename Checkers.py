# In this program computer play checkers against itself with minmax algorithm
# and get better step by step, also we can change the depth of minmax tree by
# changing "DEPTH" parameter to 1 or 2.
import numpy as np
import random
import time
import os

LR = 0.001
EPOCHS = 10
DEPTH = 2


def clear():
    os.system('clear')


def init_board():
    """
    Initialize board of checkers with given shape.
    """
    total = np.indices((8, 8)).sum(axis=0) % 2  # set all values to 1
    whites = total[:3, :]  # split whites
    middle = total[3:5, :]  # split middles
    middle[:, :] = 0  # set all middle values to 0
    blacks = total[5:, :]  # split blacks
    blacks[blacks == 1] = 2  # set all 1 values to 2 in blacks

    return np.vstack((whites, middle, blacks))  # stack all pieces vertically


def init_W():
    """
    Initialize W0 to W6.
    """
    W0 = random.randint(-100, 100)  # W0 is bias
    W1 = random.randint(0, 100)  # W1 is weight for number of white pieces
    W2 = random.randint(-100, 0)  # W2 is weight for number of black pieces
    W3 = random.randint(0, 100)  # W3 is weight for number of white kings
    W4 = random.randint(-100, 0)  # W4 is weight for number of black kings
    W5 = random.randint(-100, 0)  # W5 is weight for number of threatened whites
    W6 = random.randint(0, 100)  # W6 is weight for number of threatened whites

    W = {'0': W0, '1': W1, '2': W2, '3': W3
        , '4': W4, '5': W5, '6': W6}

    return W


def count_threatened(board, player):
    """
    Count the number of threatened pieces for specific player.
    """
    threat = 0  # initialize the number of threat for player
    length = board.shape[0]  # get length of board
    cboard = board.copy()

    # Set the numbers for pieces and kings for each player
    if player == 'white':
        piece, king = 1, 3
        op_piece, op_king = 2, 4

    elif player == 'black':
        cboard = np.flip(cboard)  # if the player is black we flip the board
        piece, king = 2, 4
        op_piece, op_king = 1, 3

    else:
        return threat

    # Count the numbers of pieces and kings and
    # stack the coordinates vertically
    index_p = np.where(cboard == piece)
    pieces = np.vstack((index_p[0], index_p[1]))
    index_k = np.where(cboard == king)
    kings = np.vstack((index_k[0], index_k[1]))
    indexes = np.hstack((pieces, kings))

    # Iterate over extracted coordinates for searching threatened pieces
    for x, y in zip(indexes[0], indexes[1]):

        if 0 < x < length - 1 and 0 < y < length - 1:  # pieces on the edge of the board cant be threatened

            # search upper left of player piece for opponent piece
            if cboard[x + 1, y - 1] in (op_piece, op_king) and cboard[x - 1, y + 1] == 0:
                threat += 1
            # search upper right of player piece for opponent piece
            if cboard[x + 1, y + 1] in (op_piece, op_king) and cboard[x - 1, y - 1] == 0:
                threat += 1
            # search lower left of player piece for opponent piece
            if cboard[x - 1, y - 1] == op_king and cboard[x + 1, y + 1] == 0:
                threat += 1
            # search lower right of player piece for opponent piece
            if cboard[x - 1, y + 1] == op_king and cboard[x + 1, y - 1] == 0:
                threat += 1

    del cboard

    return threat


def extract_x(board):
    """
    Retrieve information from board and return x's
    """
    X1 = np.count_nonzero(board == 1)  # X1 is number of white pieces
    X2 = np.count_nonzero(board == 2)  # X2 is number of black pieces
    X3 = np.count_nonzero(board == 3)  # X3 is number of white kings
    X4 = np.count_nonzero(board == 4)  # X4 is number of black kings
    X5 = count_threatened(board, 'white')  # X5 is number of threatened white pieces
    X6 = count_threatened(board, 'black')  # X6 is number of threatened black pieces

    X = {'1': X1, '2': X2, '3': X3
        , '4': X4, '5': X5, '6': X6}

    return X


def calculate_V(board, W):
    """
    Calculate V for the given status of game board.
    """
    X = extract_x(board)  # extract required X's from board

    W0, W1, W2, W3, W4, W5, W6 = W['0'], W['1'], W['2'], W['3'], W['4'], W['5'], W['6']

    X1, X2, X3, X4, X5, X6 = X['1'], X['2'], X['3'], X['4'], X['5'], X['6']

    # Calculate heuristic evaluation
    V = W0 + W1 * X1 + W2 * X2 + W3 * X3 + W4 * X4 + W5 * X5 + W6 * X6
    V = V * 0.01

    return V


def advance(arr, cor1, cor2, length, king):
    """
    Function for advance a piece in board.
    """
    temp = arr.copy()  # copy board for preventing reference change

    # Get position of two given coordinates
    val1 = temp[cor1[0], cor1[1]]
    val2 = temp[cor2[0], cor2[1]]

    # Check of the destination is end of the board or not
    # if it is end of the board the piece will transform to king
    if cor2[0] == length - 1 and val1 != king:
        val1 = king

    # Change position of two given coordinates
    temp[cor1[0], cor1[1]] = val2
    temp[cor2[0], cor2[1]] = val1

    return temp


def attack(arr, cor1, cor_op, cor2, length, king):
    """
    Function for attack and eat opponent piece.
    """
    temp = arr.copy()  # copy board for preventing reference change

    # Get position of two given coordinates
    val1 = temp[cor1[0], cor1[1]]
    val2 = temp[cor2[0], cor2[1]]

    # Check of the destination is end of the board or not
    # if it is end of the board the piece will transform to king
    if cor2[0] == length - 1:
        val1 = king

    # Change position of two given coordinates
    temp[cor1[0], cor1[1]] = val2
    temp[cor2[0], cor2[1]] = val1
    temp[cor_op[0], cor_op[1]] = 0

    return temp


def pieces_moves(pieces, length, cboard, king, op_piece, op_king):
    """
    Save moves for every piece.
    """
    moves = []

    # Iterate over pieces and search for advance and attack options
    for x, y in zip(pieces[0], pieces[1]):

        if 0 <= y < length - 1 and 0 <= x < length - 1:
            # Check lower right for advancing, if its possible add an updated board
            # with that move to the move list
            if cboard[x + 1, y + 1] == 0:
                temp = advance(cboard, (x, y), (x + 1, y + 1), length, king)
                moves.append(temp)
                del temp

            # Check lower right for attacking, if its possible add an updated board
            # with that move to the move list
            elif 0 <= y < length - 2 and 0 <= x < length - 2:
                if cboard[x + 1, y + 1] in (op_piece, op_king) and cboard[x + 2, y + 2] == 0:
                    temp = attack(cboard, (x, y), (x + 1, y + 1), (x + 2, y + 2), length, king)
                    moves.append(temp)
                    del temp

        if 0 < y <= length - 1 and 0 <= x < length - 1:
            # Check lower left for advancing, if its possible add an updated board
            # with that move to the move list
            if cboard[x + 1, y - 1] == 0:
                temp = advance(cboard, (x, y), (x + 1, y - 1), length, king)
                moves.append(temp)
                del temp

            # Check lower left for attacking, if its possible add an updated board
            # with that move to the move list
            elif 1 < y <= length - 1 and 0 <= x < length - 2:
                if cboard[x + 1, y - 1] in (op_piece, op_king) and cboard[x + 2, y - 2] == 0:
                    temp = attack(cboard, (x, y), (x + 1, y - 1), (x + 2, y - 2), length, king)
                    moves.append(temp)
                    del temp

    return moves


def kings_moves(kings, length, cboard, king, op_piece, op_king):
    """
    Save moves for every king.
    """
    moves = []

    # Iterate over kings and search for advance and attack options
    for x, y in zip(kings[0], kings[1]):

        if 0 <= y < length - 1 and 0 <= x < length - 1:
            # Check lower right for advancing, if its possible add an updated board
            # with that move to the move list
            if cboard[x + 1, y + 1] == 0:
                temp = advance(cboard, (x, y), (x + 1, y + 1), length, king)
                moves.append(temp)
                del temp

            # Check lower right for attacking, if its possible add an updated board
            # with that move to the move list
            elif 0 <= y < length - 2 and 0 <= x < length - 2:
                if cboard[x + 1, y + 1] in (op_piece, op_king) and cboard[x + 2, y + 2] == 0:
                    temp = attack(cboard, (x, y), (x + 1, y + 1), (x + 2, y + 2), length, king)
                    moves.append(temp)
                    del temp

        if 0 < y <= length - 1 and 0 <= x < length - 1:
            # Check lower left for advancing, if its possible add an updated board
            # with that move to the move list
            if cboard[x + 1, y - 1] == 0:
                temp = advance(cboard, (x, y), (x + 1, y - 1), length, king)
                moves.append(temp)
                del temp

            # Check lower left for attacking, if its possible add an updated board
            # with that move to the move list
            elif 1 < y <= length - 2 and 0 <= x < length - 2:
                if cboard[x + 1, y - 1] in (op_piece, op_king) and cboard[x + 2, y - 2] == 0:
                    temp = attack(cboard, (x, y), (x + 1, y - 1), (x + 2, y - 2), length, king)
                    moves.append(temp)
                    del temp

        if 0 <= y < length - 1 and 0 <= x < length - 1:
            # Check upper right for advancing, if its possible add an updated board
            # with that move to the move list
            if cboard[x - 1, y + 1] == 0:
                temp = advance(cboard, (x, y), (x - 1, y + 1), length, king)
                moves.append(temp)
                del temp

            # Check upper right for attacking, if its possible add an updated board
            # with that move to the move list
            elif 0 <= y < length - 2 and 0 <= x < length - 2:
                if cboard[x - 1, y + 1] in (op_piece, op_king) and cboard[x - 2, y + 2] == 0:
                    temp = attack(cboard, (x, y), (x - 1, y + 1), (x - 2, y + 2), length, king)
                    moves.append(temp)
                    del temp

        if 0 < y <= length - 1 and 0 < x <= length - 1:
            # Check upper left for advancing, if its possible add an updated board
            # with that move to the move list
            if cboard[x - 1, y - 1] == 0:
                temp = advance(cboard, (x, y), (x - 1, y - 1), length, king)
                moves.append(temp)
                del temp

            # Check upper left for attacking, if its possible add an updated board
            # with that move to the move list
            elif 1 < y <= length - 2 and 0 < x <= length - 2:
                if cboard[x - 1, y - 1] in (op_piece, op_king) and cboard[x - 2, y - 2] == 0:
                    temp = attack(cboard, (x, y), (x - 1, y - 1), (x - 2, y - 2), length, king)
                    moves.append(temp)
                    del temp

    return moves


def extract_moves(board, player):
    """
    Extract all moves a player can do.
    """
    cboard = board.copy()  # copy board to prevent reference change
    moves = []
    length = board.shape[0]  # get length of board

    if player == 'white':
        piece, king = 1, 3
        op_piece, op_king = 2, 4

    elif player == 'black':
        cboard = np.flip(cboard)  # if the player is black we flip the board
        piece, king = 2, 4
        op_piece, op_king = 1, 3

    else:
        return moves

    # Extract coordinates of pieces and kings
    index_p = np.where(cboard == piece)
    pieces = np.vstack((index_p[0], index_p[1]))
    index_k = np.where(cboard == king)
    kings = np.vstack((index_k[0], index_k[1]))

    # Extract moves for every piece
    p_moves = pieces_moves(pieces, length, cboard, king, op_piece, op_king)

    # Extract moves for every king
    k_moves = kings_moves(kings, length, cboard, king, op_piece, op_king)

    moves = k_moves + p_moves  # gather all moves in a single list

    if player == 'black':
        final_list = [np.flip(item) for item in moves]  # if player is black we flip boards
    else:
        final_list = moves

    return final_list


def minmax(moves, W, player):
    """
    Choose the best move from possible moves base on heuristic evaluation.
    """
    V_list = []

    # Iterate over possible moves and Calculate V for each one.
    for iboard in moves:
        V = calculate_V(iboard, W)
        V_list.append(V)

    if not V_list:
        return [], 0

    # Based on player turn we choose best V for play, if its white turn we choose
    # maximum value from our V list and if its black turn we choose minimum value.
    V_list = np.array(V_list)
    ar = np.argmax(V_list) if player == 'white' else np.argmin(V_list)
    board = moves[ar]
    new_V = V_list[ar]

    return board, new_V


def update_W(W, new_V, prev_V, lr, f):
    """
    Update parameters.
    """

    new_W = {}

    # Calculate error
    E = new_V - prev_V

    # Update each parameter by:
    # w_new = w_old + [(learning rate) * (num pieces) * (calculated error) * 0.01]
    # 0.01 is for prevent weights becoming very big
    for key, value in W.items():
        new_W[key] = value + (lr * f * E * 0.01)

    return new_W


def Game(W, lr, depth, verboos=1):
    """
    Function for playing a whole game and determine a winner.
    """
    board = init_board()  # init first state of the board

    whites, blacks = 12, 12

    iteration = 1
    prev_V = 0
    prev_board = np.zeros((12, 12))
    move5 = np.zeros((12, 12))
    move6 = np.zeros((12, 12))

    # # Create a list with fixed size to check if the play is draw or not
    # save_list = collections.deque(maxlen=6)
    # prev_save_list = []

    # Play until the number of pieces for one side is 0
    while whites * blacks != 0 and iteration <= 100:

        # black start the game
        if iteration % 2:
            player = 'black'
            f = blacks
        else:
            player = 'white'
            f = whites

        moves = extract_moves(board, player)  # extract possible moves

        # if not moves and whites * blacks != 0:
        #     return W, 'draw'

        if depth == 2:

            second_Vs = []
            for ind, move in enumerate(moves):
                second_moves = extract_moves(move, player)
                _, second_new_V = minmax(second_moves, W, player)
                second_Vs.append(second_new_V)

            if second_Vs:
                ar = np.argmax(second_Vs) if player == 'white' else np.argmin(second_Vs)
                board = moves[ar]
                new_V = second_Vs[ar]
            # else:
            #     board, new_V = minimax(moves, W, player)

        else:
            board, new_V = minmax(moves, W, player)  # update board with the best move

        if verboos == 2:
            print(board)
            time.sleep(2)
            print('\n'*80)

        if not iteration % 5:
            move5 = board
        if not iteration % 6:
            move6 = board
        else:
            if np.array_equal(prev_board, move5) and np.array_equal(board, move6):
                text = 'Draw'
                W = update_W(W, 0, prev_V, lr, blacks)
                return text, W

        # save_list.append(board)

        W = update_W(W, new_V, prev_V, f, lr)  # update parameters

        prev_V = new_V
        prev_board = board

        iteration += 1

        whites = np.count_nonzero(board == 1) + np.count_nonzero(board == 3)
        blacks = np.count_nonzero(board == 2) + np.count_nonzero(board == 4)

    # Update parameters for last time when a player wins
    if blacks == 0:
        text = "White is winner!"
        W = update_W(W, 100, prev_V, lr, whites)
    elif whites == 0:
        text = "Black is winner!"
        W = update_W(W, -100, prev_V, lr, blacks)
    else:
        text = 'Draw'
        W = update_W(W, 0, prev_V, lr, blacks)

    return W, text


if __name__ == '__main__':
    Wi = init_W()

    for i in range(EPOCHS):
        W, t = Game(Wi, LR, DEPTH, 2)

        print(t)
