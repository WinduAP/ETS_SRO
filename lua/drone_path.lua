
sim = require('sim')

function sysCall_init()
    droneHandle = sim.getObject('.')
    baseHandle = sim.getObject('/Quadcopter/base', {noError = true})
    if not baseHandle then
        baseHandle = sim.getObject('../base', {noError = true})
    end
    if not baseHandle then
        baseHandle = droneHandle
    end

    local pathFlags = sim.drawing_linestrip + sim.drawing_cyclic
    pathHandle = sim.addDrawingObject(pathFlags, 4, 0.0, -1, 50000, {0.0, 0.85, 1.0})

    local shadowFlags = sim.drawing_linestrip + sim.drawing_cyclic
    shadowHandle = sim.addDrawingObject(shadowFlags, 2, 0.0, -1, 50000, {0.12, 0.12, 0.15})

    lastPos = nil
    totalDistance = 0.0
    minDistanceThreshold = 0.04

    sim.addLog(sim.verbosity_scriptinfos, 'drone_path.lua: initialized successfully')
end

function sysCall_sensing()
    if not baseHandle or baseHandle == -1 then return end

    local currentPos = sim.getObjectPosition(baseHandle)

    if not lastPos then
        lastPos = {currentPos[1], currentPos[2], currentPos[3]}
        if pathHandle then
            sim.addDrawingObjectItem(pathHandle, currentPos)
        end
        if shadowHandle then
            sim.addDrawingObjectItem(shadowHandle, {currentPos[1], currentPos[2], 0.015})
        end
        return
    end

    local dx = currentPos[1] - lastPos[1]
    local dy = currentPos[2] - lastPos[2]
    local dz = currentPos[3] - lastPos[3]
    local dist = math.sqrt(dx*dx + dy*dy + dz*dz)

    if dist >= minDistanceThreshold then
        totalDistance = totalDistance + dist
        lastPos = {currentPos[1], currentPos[2], currentPos[3]}

        if pathHandle then
            sim.addDrawingObjectItem(pathHandle, currentPos)
        end
        if shadowHandle then
            sim.addDrawingObjectItem(shadowHandle, {currentPos[1], currentPos[2], 0.015})
        end
    end

end

function sysCall_cleanup()
    if pathHandle then
        sim.removeDrawingObject(pathHandle)
        pathHandle = nil
    end
    if shadowHandle then
        sim.removeDrawingObject(shadowHandle)
        shadowHandle = nil
    end
end

function clearPath()
    if pathHandle then
        sim.addDrawingObjectItem(pathHandle, nil)
    end
    if shadowHandle then
        sim.addDrawingObjectItem(shadowHandle, nil)
    end
    lastPos = nil
    totalDistance = 0.0
end

function getPathStats()
    local alt = 0.0
    if baseHandle and baseHandle ~= -1 then
        local p = sim.getObjectPosition(baseHandle)
        alt = p[3]
    end
    return totalDistance, alt
end
