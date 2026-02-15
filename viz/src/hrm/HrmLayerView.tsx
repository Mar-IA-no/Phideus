'use client';

import React, { useContext, useEffect, useRef, useState } from 'react';
import s from './HrmLayerView.module.scss';
import { fetchFontAtlasData, IFontAtlasData } from '@/src/llm/render/fontRender';
import { IRenderView } from '@/src/llm/render/modelRender';
import { useScreenLayout } from '@/src/utils/layout';
import { Resizer } from '@/src/utils/Resizer';
import { Vec3 } from '@/src/utils/vector';
import { clamp } from '@/src/utils/data';
import { useGlobalDrag, useTouchEvents } from '@/src/utils/pointer';
import { IHrmProgramState, initHrmProgramState, runHrmProgram } from './HrmProgram';
import { HrmProgramStateContext, HrmSidebar } from './HrmSidebar';
import { KeyboardManagerContext, KeyboardOrder, useGlobalKeyboard } from '@/src/utils/keyboard';
import { jumpHrmPhase } from './walkthrough/HrmWalkthrough';

class HrmCanvasRender {
    progState: IHrmProgramState;
    prevTime: number = performance.now();
    rafHandle: number = 0;
    isDirty = false;
    stopped = false;
    canvasSizeDirty = true;

    constructor(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData) {
        let state = initHrmProgramState(canvasEl, fontAtlasData);
        if (!state) throw new Error('Failed to init WebGL2');
        this.progState = state;
        this.progState.markDirty = this.markDirty;
        this.progState.walkthrough.markDirty = this.markDirty;
    }

    markDirty = () => {
        if (this.stopped) return;
        this.isDirty = true;
        if (!this.rafHandle) {
            this.prevTime = performance.now();
            this.rafHandle = requestAnimationFrame(this.loop);
        }
    };

    loop = (time: number) => {
        if (this.stopped) { this.rafHandle = 0; return; }
        this.rafHandle = requestAnimationFrame(this.loop);
        if (!this.isDirty) return;
        this.isDirty = false;
        let dt = time - this.prevTime;
        this.prevTime = time;
        if (dt < 8) dt = 16;
        this.render(time, dt);
    };

    render(time: number, dt: number) {
        let { progState } = this;
        let canvasEl = progState.render.canvasEl;

        if (this.canvasSizeDirty) {
            let bcr = canvasEl.getBoundingClientRect();
            let dpr = window.devicePixelRatio || 1;
            let w = Math.round(bcr.width * dpr);
            let h = Math.round(bcr.height * dpr);
            if (canvasEl.width !== w || canvasEl.height !== h) {
                canvasEl.width = w;
                canvasEl.height = h;
            }
            progState.render.size = new Vec3(bcr.width, bcr.height);
            this.canvasSizeDirty = false;
        }

        let view: IRenderView = { time, dt, markDirty: this.markDirty };
        runHrmProgram(view, progState);
        progState.htmlSubs.notify();
    }

    stop() {
        this.stopped = true;
        if (this.rafHandle) { cancelAnimationFrame(this.rafHandle); this.rafHandle = 0; }
    }
}

export function HrmLayerView() {
    let [canvasEl, setCanvasEl] = useState<HTMLCanvasElement | null>(null);
    let [canvasRender, setCanvasRender] = useState<HrmCanvasRender | null>(null);
    let [fontAtlasData, setFontAtlasData] = useState<IFontAtlasData | null>(null);
    let layout = useScreenLayout();
    let keyboardManager = useContext(KeyboardManagerContext);

    useGlobalKeyboard(KeyboardOrder.MainPage, (ev: KeyboardEvent) => {
        if (!canvasRender?.progState) return;
        let wt = canvasRender.progState.walkthrough;
        if (ev.key === ' ') {
            if (wt.time >= wt.phaseLength) { jumpHrmPhase(wt, 1); wt.time = 0; }
            else { wt.running = !wt.running; }
            canvasRender.markDirty();
            ev.preventDefault();
        }
        if (ev.key === 'Backspace' || ev.key === 'Delete') {
            wt.running = false; wt.time = 0;
            canvasRender.markDirty();
        }
    });

    useEffect(() => {
        document.addEventListener('keydown', keyboardManager.handleKey);
        return () => { document.removeEventListener('keydown', keyboardManager.handleKey); };
    }, [keyboardManager]);

    useEffect(() => {
        let stale = false;
        async function loadFonts() {
            let data = await fetchFontAtlasData();
            if (stale) return;
            setFontAtlasData(data);
        }
        loadFonts();
        return () => { stale = true; };
    }, []);

    useEffect(() => {
        if (!canvasEl || !fontAtlasData) return;
        let render = new HrmCanvasRender(canvasEl, fontAtlasData);
        setCanvasRender(render);
        render.markDirty();
        let resizeObserver = new ResizeObserver(() => {
            render.canvasSizeDirty = true;
            render.markDirty();
        });
        resizeObserver.observe(canvasEl);
        return () => { resizeObserver.disconnect(); render.stop(); setCanvasRender(null); };
    }, [canvasEl, fontAtlasData]);

    useEffect(() => {
        if (canvasRender) {
            canvasRender.progState.pageLayout = layout;
            canvasRender.markDirty();
        }
    }, [canvasRender, layout]);

    let sidebar = canvasRender && <div className={s.sidebar}>
        <HrmProgramStateContext.Provider value={canvasRender.progState}>
            <HrmSidebar />
        </HrmProgramStateContext.Provider>
    </div>;

    let mainView = <div className={s.canvasWrap}>
        <canvas className={s.canvas} ref={setCanvasEl} />
        {canvasRender && <HrmProgramStateContext.Provider value={canvasRender.progState}>
            <HrmCanvasEventSurface />
        </HrmProgramStateContext.Provider>}
    </div>;

    return <div className={s.view}>
        <Resizer id={"hrm-sidebar"} className={"flex-1"} vertical={!layout.isDesktop} defaultFraction={0.35}>
            {layout.isDesktop && sidebar}
            {mainView}
            {!layout.isDesktop && sidebar}
        </Resizer>
    </div>;
}

const HrmCanvasEventSurface: React.FC = () => {
    let [eventSurfaceEl, setEventSurfaceEl] = useState<HTMLDivElement | null>(null);
    let progState = React.useContext(HrmProgramStateContext);

    function pan(initial: { camAngle: Vec3, camTarget: Vec3 }, dx: number, dy: number) {
        let camAngle = initial.camAngle;
        let target = initial.camTarget.clone();
        target.z = target.z + dy * 0.1 * camAngle.z;
        let sideMul = Math.sin(camAngle.x * Math.PI / 180) > 0 ? 1 : -1;
        target.x = target.x + sideMul * dx * 0.1 * camAngle.z;
        progState.camera.center = target;
        progState.markDirty();
    }

    function rotate(initial: { camAngle: Vec3, camTarget: Vec3 }, dx: number, dy: number) {
        let camAngle = initial.camAngle.clone();
        camAngle.x = camAngle.x - dx * 0.5;
        camAngle.y = clamp(camAngle.y + dy * 0.5, -87, 87);
        progState.camera.angle = camAngle;
        progState.markDirty();
    }

    let [dragStart, setDragStart] = useGlobalDrag<{ camAngle: Vec3, camTarget: Vec3 }>(function handleMove(ev, ds) {
        let dx = ev.clientX - ds.clientX;
        let dy = ev.clientY - ds.clientY;
        if (!ds.shiftKey && !(ds.button === 1 || ds.button === 2)) pan(ds.data, dx, dy);
        else rotate(ds.data, dx, dy);
        ev.preventDefault();
    });

    useTouchEvents(eventSurfaceEl, { camAngle: progState.camera.angle, camTarget: progState.camera.center }, { alwaysSendDragEvent: true },
        function handle1PointDrag(ev, ds) {
            let dx = ev.touches[0].clientX - ds.touches[0].clientX;
            let dy = ev.touches[0].clientY - ds.touches[0].clientY;
            pan(ds.data, dx, dy);
            ev.preventDefault();
        }, function handle2PointDrag(ev, ds) {
            let dsMidX = (ds.touches[0].clientX + ds.touches[1].clientX) / 2;
            let dsMidY = (ds.touches[0].clientY + ds.touches[1].clientY) / 2;
            let evMidX = (ev.touches[0].clientX + ev.touches[1].clientX) / 2;
            let evMidY = (ev.touches[0].clientY + ev.touches[1].clientY) / 2;
            rotate(ds.data, evMidX - dsMidX, evMidY - dsMidY);
            let dsDist = Math.sqrt((ds.touches[0].clientX - ds.touches[1].clientX) ** 2 + (ds.touches[0].clientY - ds.touches[1].clientY) ** 2);
            let evDist = Math.sqrt((ev.touches[0].clientX - ev.touches[1].clientX) ** 2 + (ev.touches[0].clientY - ev.touches[1].clientY) ** 2);
            let camAngle = ds.data.camAngle.clone();
            camAngle.z = clamp(camAngle.z / (evDist / dsDist), 0.1, 100000);
            progState.camera.angle = camAngle;
            progState.markDirty();
            ev.preventDefault();
        });

    function handleMouseDown(ev: React.MouseEvent) {
        setDragStart(ev, { camAngle: progState.camera.angle, camTarget: progState.camera.center });
    }

    function handleMouseMove(ev: React.MouseEvent) {
        let canvasBcr = progState.render.canvasEl.getBoundingClientRect();
        progState.mouse.mousePos = new Vec3(ev.clientX - canvasBcr.left, ev.clientY - canvasBcr.top, 0);
        progState.markDirty();
    }

    function handleWheel(ev: React.WheelEvent) {
        let camAngle = progState.camera.angle;
        progState.camera.angle = new Vec3(camAngle.x, camAngle.y, clamp(camAngle.z * Math.pow(1.0013, ev.deltaY), 0.01, 100000));
        progState.markDirty();
        ev.stopPropagation();
    }

    if (!progState.render) return null;

    return <div
        ref={setEventSurfaceEl}
        className={s.canvasEventSurface}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onWheel={handleWheel}
        onContextMenu={ev => ev.preventDefault()}
        style={{ cursor: dragStart ? 'grabbing' : 'grab' }}
    />;
};
