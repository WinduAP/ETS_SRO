
sim = require('sim')

function sysCall_init()
    targetHandle = sim.getObject('/target', {noError = true})
    if not targetHandle then
        targetHandle = sim.getObject('/Quadcopter/target', {noError = true})
    end
    if not targetHandle then
        targetHandle = sim.getObject('.')
    end

    moveSpeed = 1.2
    yawSpeed = 1.0
    targetPos = nil
    targetYaw = 0.0

    sim.addLog(sim.verbosity_scriptinfos, 'drone_controller.lua: initialized successfully')
end

function sysCall_actuation()
    local missionActive = sim.getInt32Signal('mission_active')
    if missionActive and missionActive == 1 then
        targetPos = nil
        return
    end

    if not targetHandle then return end

    if not targetPos then
        targetPos = sim.getObjectPosition(targetHandle)
        local euler = sim.getObjectOrientation(targetHandle)
        targetYaw = euler[3]
    end

    local dt = sim.getSimulationTimeStep()
    local moved = false

    local msg, aux1, aux2 = sim.getSimulatorMessage()
    while msg ~= -1 do
        if msg == sim.message_keypress and aux1 then
            local key = aux1
            local cosY = math.cos(targetYaw)
            local sinY = math.sin(targetYaw)

            if key == 2007 then
                targetPos[1] = targetPos[1] + cosY * moveSpeed * dt * 5.0
                targetPos[2] = targetPos[2] + sinY * moveSpeed * dt * 5.0
                moved = true
            elseif key == 2008 then
                targetPos[1] = targetPos[1] - cosY * moveSpeed * dt * 5.0
                targetPos[2] = targetPos[2] - sinY * moveSpeed * dt * 5.0
                moved = true
            elseif key == 2009 then
                targetPos[1] = targetPos[1] - sinY * moveSpeed * dt * 5.0
                targetPos[2] = targetPos[2] + cosY * moveSpeed * dt * 5.0
                moved = true
            elseif key == 2010 then
                targetPos[1] = targetPos[1] + sinY * moveSpeed * dt * 5.0
                targetPos[2] = targetPos[2] - cosY * moveSpeed * dt * 5.0
                moved = true
            elseif key == 119 or key == 87 then
                targetPos[3] = targetPos[3] + moveSpeed * dt * 5.0
                moved = true
            elseif key == 115 or key == 83 then
                targetPos[3] = math.max(0.2, targetPos[3] - moveSpeed * dt * 5.0)
                moved = true
            elseif key == 97 or key == 65 then
                targetYaw = targetYaw + yawSpeed * dt * 5.0
                moved = true
            elseif key == 100 or key == 68 then
                targetYaw = targetYaw - yawSpeed * dt * 5.0
                moved = true
            end
        end
        msg, aux1, aux2 = sim.getSimulatorMessage()
    end

    if moved and targetHandle then
        targetPos[1] = math.max(-5.8, math.min(5.8, targetPos[1]))
        targetPos[2] = math.max(-5.8, math.min(5.8, targetPos[2]))
        targetPos[3] = math.max(0.2, math.min(6.5, targetPos[3]))

        sim.setObjectPosition(targetHandle, targetPos)
        sim.setObjectOrientation(targetHandle, {0, 0, targetYaw})
    end
end
