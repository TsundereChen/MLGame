"""
The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameStatus, PlatformAction
)

"""
Custom import
"""

import random

"""
Custom variable
"""
DEBUG = False

def ml_loop():
    """
    The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_served = False

    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()

        ## Initial previous ball location variable
        if not ball_served:
            ball_previous = (0,0)

        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            ball_served = False

            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information

        # Find the ball
        ball_location = scene_info.ball

        # We only want to calculate the destination of the ball after the ball is served
        # It's meanless to calculate the value if the ball is still on the platform
        if ball_served:
            ball_destination = ballDestination(ball_location, ball_previous)

        # Update ball_previous location for next move
        ball_previous = ball_location

        # 3.4. Send the instruction for this frame to the game process
        if not ball_served:
            LEFT_OR_RIGHT = random.randint(0,1)
            if(LEFT_OR_RIGHT):
                comm.send_instruction(scene_info.frame, PlatformAction.SERVE_TO_RIGHT)
            else:
                comm.send_instruction(scene_info.frame, PlatformAction.SERVE_TO_LEFT)
            ball_served = True
        else:
            if ball_destination == (-1, -1):
                comm.send_instruction(scene_info.frame, PlatformAction.NONE)
            else:
                # Add offset to platform to locate center of platform
                platformLocationX = scene_info.platform[0] + 20
                platformLocationY = scene_info.platform[1]
                distancePlatformDestionation = platformLocationX - ball_destination[0]
                if DEBUG:
                    print("DEBUG: PlatformLocation: ({}, {})".format(platformLocationX,platformLocationY))
                    print("DEBUG: Prediction: ({}, {})".format(ball_destination[0],ball_destination[1]))
                if(abs(distancePlatformDestionation) < 15):
                    # The platform should be able to catch the ball
                    # Don't move the platform
                    comm.send_instruction(scene_info.frame, PlatformAction.NONE)
                elif distancePlatformDestionation > 0:
                    # Platform is at RHS of the destination
                    # Move platform to the left
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                else:
                    # Platform is at LHS of the destination
                    # Move platform to the right
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)


def ballDestination(currentLocation, previousLocation):
    """
    ballDestination(currentLocation, previousLocation)

        currentLocation: tuple, stands for ball location (x, y)
        previousLocation: tuple, stands for ball location (x, y)

        return: tuple, stands for our prediction for the destination of the ball
    """
    curX = currentLocation[0]
    curY = currentLocation[1]
    preX = previousLocation[0]
    preY = previousLocation[1]
    deltaX = curX - preX
    deltaY = curY - preY
    if DEBUG:
        print("DEBUG: currentLocation: ({}, {})".format(curX, curY))
        print("DEBUG: previousLocation: ({}, {})".format(preX, preY))
        print("DEBUG: delta: ({}, {})".format(deltaX, deltaY))
    returnValue = (-1,-1)
    if(deltaY > 0):
        # Moving downward
        # Check if the ball is not at the top of the map
        # We only calculate when the requirement is meet
        if curY > 150:
            while(curY < 400):
                # Emulate the ball movement
                curX = curX + deltaX
                curY = curY + deltaY
            # The curY should be around 400-ish
            # Now check value of curX
            if curX < 0:
                curX = -curX
            elif curX > 200:
                curX = 400 - curX
            returnValue = (curX, curY)
    return returnValue
