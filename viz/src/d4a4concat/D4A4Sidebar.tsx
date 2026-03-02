'use client';

import React, { createContext, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import s from '@/src/llm/Commentary.module.scss';
import ps from '@/src/phideus/PhideusSidebar.module.scss';
import { ID4A4ProgramState } from './D4A4Program';
import { ID4A4PhaseDef, jumpD4A4Phase, phaseToD4A4Group, D4A4Phase } from './walkthrough/D4A4Walkthrough';
import { useRequestAnimationFrame, useSubscriptions } from '@/src/utils/hooks';
import { ProgramStateContext } from '@/src/llm/Sidebar';
import { walkthroughToParagraphs } from '@/src/llm/Commentary';
import { eventEndTime, ICommentary, isCommentary, ITimeInfo } from '@/src/llm/walkthrough/WalkthroughTools';
import { lerpSmoothstep } from '@/src/utils/math';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronLeft, faChevronRight } from '@fortawesome/free-solid-svg-icons';

export let D4A4ProgramStateContext = createContext<ID4A4ProgramState>(null!);

export function useD4A4ProgramState() {
    let context = useContext(D4A4ProgramStateContext);
    useSubscriptions(context?.htmlSubs);
    return context;
}

interface INode {
    commentary?: ICommentary;
    times?: ITimeInfo[];
    isBreak: boolean;
    start: number;
    end: number;
}

export const D4A4Sidebar: React.FC = () => {
    let progState = useD4A4ProgramState();
    let [parasEl, setParasEl] = useState<HTMLDivElement | null>(null);
    let wt = progState.walkthrough;

    function handleKeyDown(ev: React.KeyboardEvent) { if (ev.key === ' ') ev.preventDefault(); }

    function handleContinueClick() {
        if (wt.time >= wt.phaseLength) { jumpD4A4Phase(wt, 1); wt.time = 0; }
        else { wt.running = !wt.running; }
        progState.markDirty();
    }

    function handleAdvanceClick() {
        if (wt.time >= wt.phaseLength) { jumpD4A4Phase(wt, 1); wt.time = 0; }
        else {
            wt.running = true;
            let node = nodes.find(n => n.end > wt.time);
            let speed = 15;
            if (node && node.end > wt.time) speed = (node.end - wt.time) * 2;
            wt.speed = speed;
        }
        progState.markDirty();
    }

    function handlePhaseDeltaClick(delta: number) { jumpD4A4Phase(wt, delta); progState.markDirty(); }

    let numTimes = wt.times.length;

    let { nodes } = useMemo(() => {
        let nodes: INode[] = [];
        let prevIsTime = false;
        for (let c of wt.times) {
            if (isCommentary(c)) {
                nodes.push({ commentary: c, isBreak: false, start: c.start, end: eventEndTime(c) });
                prevIsTime = false;
            } else {
                !prevIsTime && nodes.push({ times: [], isBreak: false, start: c.start, end: c.start });
                let lastNode = nodes[nodes.length - 1];
                lastNode.times!.push(c);
                lastNode.isBreak ||= !!c.isBreak;
                lastNode.end = eventEndTime(c);
                prevIsTime = true;
            }
        }
        return { nodes };
    }, [wt.times]);

    let { prevBreak, nextBreak } = useMemo(() => {
        let prevBreak = -1, nextBreak = -1, lastBreak = -1;
        for (let i = 0; i < nodes.length + 1; i++) {
            let node = nodes[i];
            if (node?.isBreak || i === nodes.length) {
                if (i === nodes.length || node.start >= wt.time) { nextBreak = lastBreak - 1; break; }
                prevBreak = lastBreak + 1;
                lastBreak = i;
            }
        }
        return { prevBreak, nextBreak };
    }, [wt.time, nodes]);

    interface IGuideLayout { width: number; height: number; parentHeight: number; childRanges: IChildRange[]; }
    interface IChildRange { top: number; bottom: number; height: number; nodeId: number; startT: number; endT: number; }

    let [guideLayout, setGuideLayout] = useState<IGuideLayout>({ width: 0, height: 0, parentHeight: 0, childRanges: [] });

    useLayoutEffect(() => {
        function handleChildren() {
            if (!parasEl?.children) return;
            let parasBcr = parasEl.getBoundingClientRect();
            let ranges: IChildRange[] = [];
            for (let child of parasEl.children) {
                let nid = parseInt(child.getAttribute('data-nid')!);
                let c = nodes[nid];
                if (!c) continue;
                let cStart = c.commentary?.start ?? c.times![0].start;
                let cEnd = eventEndTime(c.commentary ?? c.times![c.times!.length - 1]);
                let childBcr = child.getBoundingClientRect();
                ranges.push({ top: childBcr.top - parasBcr.top, bottom: childBcr.bottom - parasBcr.top, nodeId: nid, startT: cStart, endT: cEnd, height: childBcr.height });
            }
            setGuideLayout({ width: parasBcr.width - 40, height: parasBcr.height, parentHeight: parasEl.parentElement!.getBoundingClientRect().height, childRanges: ranges });
        }
        if (parasEl) {
            let observer = new ResizeObserver(handleChildren);
            observer.observe(parasEl);
            observer.observe(parasEl.parentElement!);
            return () => { observer.disconnect(); };
        }
    }, [nodes, parasEl, wt.phase, numTimes]);

    let { rangeInfo, currPos } = useMemo(() => {
        let rangeInfo = { start: 0, end: 0, width: 1 };
        let currPos = 0;
        for (let range of guideLayout.childRanges) {
            if (range.startT <= wt.time && range.endT >= wt.time) { currPos = range.bottom; break; }
        }
        let startPos = 0, endPos = 0;
        function findChild(nid: number) { return guideLayout.childRanges.find(c => c.nodeId === nid); }
        if (nodes.length > 0) { let child = findChild(Math.max(0, prevBreak)); if (child) startPos = child.top; }
        if (nextBreak >= 0) { let child = findChild(nextBreak); if (child) endPos = child.bottom; }
        rangeInfo = { start: startPos, end: endPos, width: guideLayout.width };
        return { rangeInfo, currPos };
    }, [wt.time, guideLayout, nodes, prevBreak, nextBreak]);

    let group = phaseToD4A4Group(wt);
    let phase = group?.phases.find(p => p.id === wt.phase)!;

    useEffect(() => { if (parasEl) parasEl.parentElement!.scrollTop = 0; }, [parasEl, wt.phase]);

    let prevPhase = useRef(-1 as number);
    let upToDate = wt.times.length > 0;

    useEffect(() => {
        if (parasEl) {
            let delta = 512;
            if (prevPhase.current !== wt.phase) prevPhase.current = wt.phase;
            else if (wt.time > 0) parasEl.parentElement!.scrollTo({ top: rangeInfo.start + delta, behavior: 'smooth' });
        }
    }, [rangeInfo.start, rangeInfo.end, currPos, parasEl, upToDate, guideLayout.height, guideLayout.parentHeight, wt.phase, wt.time]);

    return <ProgramStateContext.Provider value={progState as any}>
        <div className={s.chapterControls}>
            <button className={clsx(s.btn, s.prevNextBtn)} onClick={() => handlePhaseDeltaClick(-1)}>
                <FontAwesomeIcon icon={faChevronLeft} />
            </button>
            <div className={s.chapterTitle}>Chapter: {phase?.title}</div>
            <button className={clsx(s.btn, s.prevNextBtn)} onClick={() => handlePhaseDeltaClick(1)}>
                <FontAwesomeIcon icon={faChevronRight} />
            </button>
        </div>
        <div className={s.walkthroughViewport}>
            <div className={s.walkthroughText} tabIndex={0} onKeyDownCapture={handleKeyDown}>
                <D4A4PhaseMenu />
                <div className={s.divider} />
                <div className={s.walkthroughParas} ref={setParasEl}>
                    {walkthroughToParagraphs(wt as any, nodes)}
                    <SectionHighlight key={nextBreak} top={rangeInfo.start} height={rangeInfo.end - rangeInfo.start} width={rangeInfo.width} />
                    {!wt.running && <>
                        <div className={s.dividerLine} style={{ top: currPos }} />
                        <SpaceToContinueHint top={currPos} onClick={handleContinueClick} />
                    </>}
                </div>
            </div>
        </div>
        <div className={s.controls}>
            <button className={clsx(s.btn, "flex-[2] bg-blue-300 border border-blue-600 hover:bg-blue-400")} onClick={handleContinueClick}>
                <div>Continue</div>
            </button>
            <button className={clsx(s.btn, "ml-4 min-w-[100px] bg-white border border-blue-600 hover:bg-blue-200")} onClick={handleAdvanceClick}>
                <div>Skip</div>
            </button>
        </div>
    </ProgramStateContext.Provider>;
};

const D4A4PhaseMenu: React.FC = () => {
    let progState = useD4A4ProgramState();
    let wt = progState.walkthrough;

    function handlePhaseClick(ev: React.MouseEvent, phase: ID4A4PhaseDef) {
        if (wt.phase !== phase.id) { wt.phase = phase.id; wt.time = 0; wt.running = false; progState.markDirty(); }
        ev.preventDefault();
    }

    return <div className={ps.tocBackground}>
        <div className={ps.tocTitle}>d4a4 — Concat Descriptor Injection</div>
        {wt.phaseList.map((group) => (
            <div key={group.groupId}>
                <div className={ps.phaseGroupTitle}>{group.title}</div>
                {group.phases.map((phase) => {
                    let active = wt.phase === phase.id;
                    return <div key={phase.id} className={clsx(ps.phase, active && ps.active)} onClick={ev => handlePhaseClick(ev, phase)}>
                        <div className={ps.phaseTitle}>{phase.title}</div>
                    </div>;
                })}
            </div>
        ))}
    </div>;
};

const SpaceToContinueHint: React.FC<{ top: number; onClick: React.MouseEventHandler }> = ({ top, onClick }) => {
    return <div className={"absolute flex justify-center pointer-events-none top-0 left-0 right-0"} style={{ top, transform: `translateY(20px)` }}>
        <div className={"flex-shrink py-2 px-4 bg-blue-200 shadow-md rounded-3xl pointer-events-auto text-black cursor-pointer"} onClick={onClick}>
             Press <span>Space</span> to continue
        </div>
    </div>;
};

const SectionHighlight: React.FC<{ top: number; height: number; width: number }> = ({ top, height, width }) => {
    let [tick, setTick] = useState(0);
    useRequestAnimationFrame(tick < 2, (dt) => { setTick(tick + dt); });
    let rectPad = 12;
    let svgW = width + rectPad * 2;
    let svgH = height + rectPad * 2;
    let pad = 3;
    let strokeWidth = lerpSmoothstep(3, 0, tick);
    if (height <= 0) return null;
    return <div className={s.sectionHighlightWrap} style={{ top: top - rectPad, height: svgH, width: svgW, left: -rectPad }}>
        <svg viewBox={`0 0 ${svgW} ${svgH}`} className={s.sectionHighlight}>
            <rect x={pad} y={pad} width={svgW - 2 * pad} height={svgH - 2 * pad} fill="none" stroke="#cc3366" strokeWidth={strokeWidth} opacity={strokeWidth} rx={5} ry={5} />
        </svg>
    </div>;
};
