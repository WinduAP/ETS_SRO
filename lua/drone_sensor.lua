
sim = require('sim')

function sysCall_init()
    mountHandle = sim.getObject('.')
    droneBaseHandle = sim.getObject('/Quadcopter/base', {noError = true})
    if not droneBaseHandle then
        droneBaseHandle = sim.getObject('..', {noError = true})
    end

    depthDown = sim.getObject('/Quadcopter/sensorMount/depthDown', {noError = true})
    if not depthDown then depthDown = sim.getObject('../depthDown', {noError = true}) end

    depthFront = sim.getObject('/Quadcopter/sensorMount/depthFront', {noError = true})
    if not depthFront then depthFront = sim.getObject('../depthFront', {noError = true}) end

    depthLeft = sim.getObject('/Quadcopter/sensorMount/depthLeft', {noError = true})
    if not depthLeft then depthLeft = sim.getObject('../depthLeft', {noError = true}) end

    depthRight = sim.getObject('/Quadcopter/sensorMount/depthRight', {noError = true})
    if not depthRight then depthRight = sim.getObject('../depthRight', {noError = true}) end

    lidarS0 = sim.getObject('/Quadcopter/sensorMount/lidar2D/s0', {noError = true})
    if not lidarS0 then lidarS0 = sim.getObject('../lidar2D/s0', {noError = true}) end

    lidarS1 = sim.getObject('/Quadcopter/sensorMount/lidar2D/s1', {noError = true})
    if not lidarS1 then lidarS1 = sim.getObject('../lidar2D/s1', {noError = true}) end

    lidarS2 = sim.getObject('/Quadcopter/sensorMount/lidar2D/s2', {noError = true})
    if not lidarS2 then lidarS2 = sim.getObject('../lidar2D/s2', {noError = true}) end

    lidarS3 = sim.getObject('/Quadcopter/sensorMount/lidar2D/s3', {noError = true})
    if not lidarS3 then lidarS3 = sim.getObject('../lidar2D/s3', {noError = true}) end

    accumulatedPoints = {}
    accumulatedSensorIds = {}
    lastScan2D = {}
    lastDronePose = {0, 0, 0, 0}

    scanEvery = 1
    stepCounter = 0
    liveCloudVisible = true
    farClipMargin = 0.99
    droneSelfRadiusSq = 0.30 * 0.30

    local cloudFlags = sim.drawing_points + sim.drawing_itemcolors + sim.drawing_cyclic
    liveCloudHandle = sim.addDrawingObject(cloudFlags, 3, 0.0, -1, 150000, {0.5, 0.5, 0.5})

    local lineFlags = sim.drawing_lines + sim.drawing_50percenttransparency + sim.drawing_cyclic
    lidarLinesHandle = sim.addDrawingObject(lineFlags, 1, 0.0, -1, 720, {0.0, 0.9, 0.9})

    sim.addLog(sim.verbosity_scriptinfos, 'drone_sensor.lua: initialized successfully')
end

function heightToColor(z, minZ, maxZ)
    local t = (z - minZ) / (maxZ - minZ)
    if t < 0.0 then t = 0.0 end
    if t > 1.0 then t = 1.0 end

    local r, g, b
    if t < 0.25 then
        local f = t / 0.25
        r = 0.0
        g = f
        b = 1.0
    elseif t < 0.50 then
        local f = (t - 0.25) / 0.25
        r = 0.0
        g = 1.0
        b = 1.0 - f
    elseif t < 0.75 then
        local f = (t - 0.50) / 0.25
        r = f
        g = 1.0
        b = 0.0
    else
        local f = (t - 0.75) / 0.25
        r = 1.0
        g = 1.0 - f
        b = 0.0
    end
    return r, g, b
end

function processDepthSensor(sensorHandle, sensorId, strideU, strideV, liveBatch)
    if not sensorHandle or sensorHandle == -1 then return end

    local depthBuf, res = sim.getVisionSensorDepth(sensorHandle, 1)
    if not depthBuf or #depthBuf == 0 then return end

    local str = (type(depthBuf) == 'string') and depthBuf or table.concat(depthBuf)
    local depths = sim.unpackFloatTable(str)
    if not depths or #depths == 0 then return end

    local rx = res[1]
    local ry = res[2]
    local m = sim.getObjectMatrix(sensorHandle)
    local fov = sim.getObjectFloatParam(sensorHandle, sim.visionfloatparam_perspective_angle)
    local tanHalfFov = math.tan(fov * 0.5)
    local aspect = ry / rx
    local farClip = sim.getObjectFloatParam(sensorHandle, sim.visionfloatparam_far_clipping)
    if not farClip or farClip <= 0 then farClip = 12.0 end
    local farThresh = farClip * farClipMargin

    local dronePos = {0, 0, 0}
    if droneBaseHandle and droneBaseHandle ~= -1 then
        dronePos = sim.getObjectPosition(droneBaseHandle)
    end

    strideU = strideU or 2
    strideV = strideV or 2

    for v = 0, ry - 1, strideV do
        for u = 0, rx - 1, strideU do
            local idx = v * rx + u + 1
            local z_cam = depths[idx]

            if z_cam and z_cam > 0.05 and z_cam < farThresh then
                local x_norm = - ( (u + 0.5) / rx - 0.5 ) * 2.0
                local y_norm = ( (v + 0.5) / ry - 0.5 ) * 2.0

                local x_cam = z_cam * tanHalfFov * x_norm
                local y_cam = z_cam * tanHalfFov * y_norm * aspect

                local x_w = m[1]*x_cam + m[2]*y_cam + m[3]*z_cam + m[4]
                local y_w = m[5]*x_cam + m[6]*y_cam + m[7]*z_cam + m[8]
                local z_w = m[9]*x_cam + m[10]*y_cam + m[11]*z_cam + m[12]

                local dx = x_w - dronePos[1]
                local dy = y_w - dronePos[2]
                local dz = z_w - dronePos[3]
                if (dx*dx + dy*dy + dz*dz) > droneSelfRadiusSq then
                    table.insert(accumulatedPoints, x_w)
                    table.insert(accumulatedPoints, y_w)
                    table.insert(accumulatedPoints, z_w)
                    table.insert(accumulatedSensorIds, sensorId)

                    if liveCloudVisible and liveBatch then
                        local r, g, b = heightToColor(z_w, 0.0, 4.2)
                        table.insert(liveBatch, x_w)
                        table.insert(liveBatch, y_w)
                        table.insert(liveBatch, z_w)
                        table.insert(liveBatch, r)
                        table.insert(liveBatch, g)
                        table.insert(liveBatch, b)
                    end
                end
            end
        end
    end
end

function processLidar2D(liveBatch, lidarLinesBatch)
    local sensors = {
        {handle = lidarS0, baseYaw = 0.0},
        {handle = lidarS1, baseYaw = math.pi * 0.5},
        {handle = lidarS2, baseYaw = math.pi},
        {handle = lidarS3, baseYaw = -math.pi * 0.5}
    }

    local dronePos = {0, 0, 0}
    local droneYaw = 0.0
    if droneBaseHandle and droneBaseHandle ~= -1 then
        dronePos = sim.getObjectPosition(droneBaseHandle)
        local droneEuler = sim.getObjectOrientation(droneBaseHandle)
        droneYaw = droneEuler[3]
    end
    lastDronePose = {dronePos[1], dronePos[2], dronePos[3], droneYaw}
    lastScan2D = {}

    for _, s in ipairs(sensors) do
        if s.handle and s.handle ~= -1 then
            local depthBuf, res = sim.getVisionSensorDepth(s.handle, 1)
            if depthBuf and #depthBuf > 0 then
                local str = (type(depthBuf) == 'string') and depthBuf or table.concat(depthBuf)
                local depths = sim.unpackFloatTable(str)
                if depths and #depths > 0 then
                    local rx = res[1]
                    local m = sim.getObjectMatrix(s.handle)
                    local farClip = sim.getObjectFloatParam(s.handle, sim.visionfloatparam_far_clipping)
                    if not farClip or farClip <= 0 then farClip = 10.0 end
                    local farThresh = farClip * farClipMargin

                    for u = 0, rx - 1 do
                        local z_cam = depths[u + 1]
                        if not (z_cam and z_cam > 0.05 and z_cam < farThresh) then
                            local xn = - ( (u + 0.5) / rx - 0.5 ) * 2.0
                            local a = s.baseYaw + math.atan2(xn, 1.0)
                            while a > math.pi do a = a - 2 * math.pi end
                            while a < -math.pi do a = a + 2 * math.pi end
                            table.insert(lastScan2D, a)
                            table.insert(lastScan2D, farClip)
                        end
                        if z_cam and z_cam > 0.05 and z_cam < farThresh then
                            local x_norm = - ( (u + 0.5) / rx - 0.5 ) * 2.0
                            local x_cam = z_cam * x_norm
                            local r = math.sqrt(x_cam * x_cam + z_cam * z_cam)
                            local rayLocalAngle = math.atan2(x_cam, z_cam)
                            local angleRelDrone = s.baseYaw + rayLocalAngle

                            while angleRelDrone > math.pi do angleRelDrone = angleRelDrone - 2 * math.pi end
                            while angleRelDrone < -math.pi do angleRelDrone = angleRelDrone + 2 * math.pi end

                            table.insert(lastScan2D, angleRelDrone)
                            table.insert(lastScan2D, r)

                            local x_w = m[1]*x_cam + m[3]*z_cam + m[4]
                            local y_w = m[5]*x_cam + m[7]*z_cam + m[8]
                            local z_w = m[9]*x_cam + m[11]*z_cam + m[12]

                            local dx = x_w - dronePos[1]
                            local dy = y_w - dronePos[2]
                            local dz = z_w - dronePos[3]
                            if (dx*dx + dy*dy + dz*dz) > droneSelfRadiusSq then
                                table.insert(accumulatedPoints, x_w)
                                table.insert(accumulatedPoints, y_w)
                                table.insert(accumulatedPoints, z_w)
                                table.insert(accumulatedSensorIds, 4)

                                if liveCloudVisible and liveBatch then
                                    local cr, cg, cb = heightToColor(z_w, 0.0, 4.2)
                                    table.insert(liveBatch, x_w)
                                    table.insert(liveBatch, y_w)
                                    table.insert(liveBatch, z_w)
                                    table.insert(liveBatch, cr)
                                    table.insert(liveBatch, cg)
                                    table.insert(liveBatch, cb)
                                end

                                if lidarLinesBatch and (u % 4 == 0) then
                                    table.insert(lidarLinesBatch, m[4])
                                    table.insert(lidarLinesBatch, m[8])
                                    table.insert(lidarLinesBatch, m[12])
                                    table.insert(lidarLinesBatch, x_w)
                                    table.insert(lidarLinesBatch, y_w)
                                    table.insert(lidarLinesBatch, z_w)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function sysCall_sensing()
    stepCounter = stepCounter + 1
    if (stepCounter % scanEvery) ~= 0 then return end

    local liveBatch = liveCloudVisible and {} or nil
    local lidarLinesBatch = {}

    processDepthSensor(depthDown, 0, 2, 2, liveBatch)

    processDepthSensor(depthFront, 1, 2, 2, liveBatch)

    processDepthSensor(depthLeft, 2, 2, 2, liveBatch)

    processDepthSensor(depthRight, 3, 2, 2, liveBatch)

    processLidar2D(liveBatch, lidarLinesBatch)

    if liveCloudVisible and liveBatch and #liveBatch > 0 and liveCloudHandle then
        sim.addDrawingObjectItem(liveCloudHandle, liveBatch)
    end

    if lidarLinesBatch and #lidarLinesBatch > 0 and lidarLinesHandle then
        sim.addDrawingObjectItem(lidarLinesHandle, lidarLinesBatch)
    end
end

function sysCall_cleanup()
    if liveCloudHandle then
        sim.removeDrawingObject(liveCloudHandle)
        liveCloudHandle = nil
    end
    if lidarLinesHandle then
        sim.removeDrawingObject(lidarLinesHandle)
        lidarLinesHandle = nil
    end
end

function getNewPoints()
    if #accumulatedPoints == 0 then
        return '', ''
    end
    local packedPts = sim.packFloatTable(accumulatedPoints)
    local packedIds = sim.packInt32Table(accumulatedSensorIds)
    accumulatedPoints = {}
    accumulatedSensorIds = {}
    return packedPts, packedIds
end

function getNewScan2D()
    local packedScan = sim.packFloatTable(lastScan2D)
    return packedScan, {lastDronePose[1], lastDronePose[2], lastDronePose[3], lastDronePose[4]}
end

function setLiveCloudVisible(visible)
    liveCloudVisible = visible
    if not visible and liveCloudHandle then
        sim.addDrawingObjectItem(liveCloudHandle, nil)
    end
end

function clearLiveCloud()
    if liveCloudHandle then
        sim.addDrawingObjectItem(liveCloudHandle, nil)
    end
    accumulatedPoints = {}
    accumulatedSensorIds = {}
end
