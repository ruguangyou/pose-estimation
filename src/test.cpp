#include <iostream>

#define WINDOWSIZE 3

int main() {
    double p_delta_pose[WINDOWSIZE*6];
    double** delta_pose;
    // double delta_pose[WINDOWSIZE][6];
    double delta_v_dbga[WINDOWSIZE][9];
    for (int i = 0; i < WINDOWSIZE; i++) {
        *delta_pose = p_delta_pose + WINDOWSIZE * i;
        for (int j = 0; j < 6; j++)
            delta_pose[i][j] = 0;
        for (int j = 0; j < 9; j++)
            delta_v_dbga[i][j] = 0;
    }

    return 0;
}