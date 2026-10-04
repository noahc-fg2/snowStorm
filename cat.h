#ifndef CAT_H
#define CAT_H


class cat{


    public:
        void detect();

    private:
        bool isFound();
        bool isMoving(int x, int y);
        bool isSleeping(int x, int y);
        int getX();
        int getY();

};

#endif // CAT_Hs