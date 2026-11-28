// Autoplayer - adaptive checkpoint-rewind input search (Geode, GD 2.2081).
//
// The adaptive training/search algorithm is intentionally unchanged. It still
// searches one button state per TICKS-sized chunk, mutates edge patterns, uses
// checkpoint rewind, and performs the same archive/sweep/regression logic.
//
// Recording/replay model:
//   - Training records one TraceFrame for every PlayerObject::update physics
//     step on the currently successful path (not once per rendered frame).
//   - Each frame stores the resulting player state, input state, and physics dt
//     for that exact player update.
//   - The reset callback used to begin a new search trial is not itself a
//     physics update, so it primes the next input but is not recorded.
//   - Replay applies each recorded input immediately before the matching
//     PlayerObject::update call and compares state immediately afterwards.
//     Player X and render-frame timing are never used to schedule input.
//   - Checkpoint promotion commits the exact trace prefix captured when that
//     checkpoint was created, preserving the already-executed tail.
//   - A gamemode/gravity section is only committed after its assembled inputs
//     survive one continuous validation pass from the section-start checkpoint.
//
// Same game calls as the working trainer mod: postUpdate, createCheckpoint,
// manual m_checkpointArray, gated resetLevel, pushButton/releaseButton.

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace geode::prelude;

$on_mod(Loaded) {
    log::info("Autoplayer: mod code loaded");
}

namespace {
    constexpr int   TICKS         = 2;     // search chunks remain 2 postUpdate calls
    constexpr int   CUBE_RAPID_GAP = 1;    // conservative: only bridge a one-chunk cube release
    constexpr int   HORIZON       = 60;    // chunks a pattern must survive
    constexpr int   COMMIT        = 20;    // earliest chunk (from layer start) a checkpoint may be taken
    constexpr int   DEADLINE      = 36;    // flying modes: fallback checkpoint deadline
    constexpr int   SHIP_CHECKPOINT_LEAD = 100; // chunks before a known ship death to place a checkpoint
    constexpr int   SHIP_CHECKPOINT_VERIFY = 8; // chunks to verify after a targeted ship checkpoint
    constexpr int   SHIP_CHECKPOINT_CLEAR = 12; // wait this far past a known death before checkpointing again
    constexpr float SHIP_MIN_TAP_LEAD = 48.f;   // let ship inputs begin well before the learned death point
    constexpr int   MAX_RUN       = HORIZON + 120;   // tap modes: longest wait for a safe moment
    constexpr int   LAYER_TRIES   = 500;   // attempts per layer before backtracking
    constexpr size_t ARCHIVE_SIZE = 12;    // best plus diverse flight trajectories kept per layer
    constexpr int   WARMUP        = 1;     // ticks to wait after the level loads
    constexpr float RUNWAY_AHEAD  = 45.f;  // no spike within this far ahead of a checkpoint
    constexpr float RUNWAY_BEHIND = 40.f;  // ...or this far behind it
    constexpr int   BAN_RADIUS    = 3;     // chunks around a dead-ended start that are avoided
    constexpr int   SAME_SPOT_TOLERANCE = 10; // treat deaths within +/- this many chunks as the same obstacle
    constexpr int   SAME_SPOT_LIMIT = 6; // repeated deaths at the same obstacle before reverting one checkpoint
    constexpr int   MAX_STRIKES   = 3;     // failed children before a layer is dropped as well
    constexpr int   SWEEP_START   = 40;    // attempts without improvement before the edge sweep starts
    constexpr size_t SWEEP_MAX    = 900;   // longer focused edge/pulse search for difficult flight sections
    constexpr float BIG_MOVE      = 0.15f; // chance an earlier/later move takes a big step
    constexpr int SHIP_STALL_DEATHS = 12;       // deaths at same obstacle before deeper rewind
    constexpr int SHIP_STALL_NEXT_THRESHOLD = 8; // additional deaths before escalating again
    constexpr int SHIP_STALL_TOLERANCE = 10;    // global chunk tolerance
    constexpr int SHIP_STALL_MAX_REGRESS = 5;   // maximum checkpoints to remove at once

    constexpr float TRACE_POSITION_TOLERANCE = 0.5f;
    constexpr float TRACE_VELOCITY_TOLERANCE = 0.5f;
    constexpr float TRACE_ROTATION_TOLERANCE = 1.0f;

    // A section is accepted only after one uninterrupted validation pass.
    // These are wider than per-frame replay diagnostics because tiny numerical
    // differences can occur without changing the route, but large drift must
    // send the section back through local training.
    constexpr float SECTION_MAX_POSITION_ERROR = 8.0f;
    constexpr float SECTION_MAX_VELOCITY_ERROR = 10.0f;
    constexpr float SECTION_MAX_ROTATION_ERROR = 35.0f;

    enum Op { OP_EARLY = 0, OP_LATE, OP_ADD, OP_REMOVE, OP_EXTEND, OP_SHRINK, OP_RANDOM, N_OPS };
    const char* const OP_NAME[N_OPS] = {"earlier", "later", "add-tap", "remove-tap",
                                        "longer", "shorter", "random"};

    struct Ind {
        std::vector<uint8_t> pat;
        int fit = 0;      // chunk where it died
        int vdir = 0;     // vertical direction at death: +1 rising, -1 falling, 0 level
        float deathY = 0.f;
        float deathVy = 0.f;
    };

    enum TraceFlags : uint8_t {
        TRACE_GROUND       = 1 << 0,
        TRACE_UPSIDE_DOWN  = 1 << 1,
        TRACE_SHIP         = 1 << 2,
        TRACE_BIRD         = 1 << 3,
        TRACE_BALL         = 1 << 4,
        TRACE_DART         = 1 << 5,
        TRACE_ROBOT        = 1 << 6,
        TRACE_SWING        = 1 << 7
    };

    // One sample per actual physics update. `input` is the button state USED
    // during this update, not the state selected after this update.
    struct TraceFrame {
        uint64_t update = 0;
        uint8_t input = 0;
        float x = 0.f;
        float y = 0.f;
        float vx = 0.f;
        float vy = 0.f;
        float rotation = 0.f;
        float dt = 0.f;
        uint8_t flags = 0;
    };

    constexpr int SWEEP_DELTAS[] = {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64};
    constexpr int N_SWEEP_DELTAS = 14;
    struct SweepMove { int edge; int delta; bool tap; };   // tap = slide press+release together

    // chunk indices where the button state flips (initial state = released)
    std::vector<int> toEdges(const std::vector<uint8_t>& p) {
        std::vector<int> e;
        uint8_t prev = 0;
        for (size_t c = 0; c < p.size(); ++c) {
            if (p[c] != prev) { e.push_back(static_cast<int>(c)); prev = p[c]; }
        }
        return e;
    }

    std::vector<uint8_t> fromEdges(const std::vector<int>& e, size_t len) {
        std::vector<uint8_t> p(len, 0);
        uint8_t s = 0;
        size_t i = 0;
        for (size_t c = 0; c < len; ++c) {
            while (i < e.size() && e[i] <= static_cast<int>(c)) { s ^= 1; ++i; }
            p[c] = s;
        }
        return p;
    }
}

class $modify(BotLayer, PlayLayer) {
    struct Layer {
        CheckpointObject* cp = nullptr;                // nullptr = level start
        size_t len = 0;                                // absolute chunk index of this checkpoint
        size_t traceLen = 0;                           // committed physics-trace length at this checkpoint
        std::vector<uint8_t> tail;                     // seed pattern for attempts from here
        std::vector<std::vector<uint8_t>> banned;      // lock-in prefixes known to dead-end
        int strikes = 0;
        int lastDeathChunk = -1;                      // most recent local chunk where this layer died
        int sameSpotDeaths = 0;                       // consecutive deaths near the same obstacle
        bool holdAtCp = false;                         // button state when the checkpoint was taken
    };

    struct Fields {
        bool practiceSet = false, started = false, engaged = false, done = false;
        bool dead = false, holding = false, allowReset = false, needRewind = true;
        bool replayFinished = false, fly = false, ship = false, wave = false;
        int warmup = 0, chunk = 0, tickIn = 0, tries = 0, rTick = 0;
        int deaths = 0, cpLogs = 0, pressLogs = 0, bestLogs = 0, vdir = 0;

        // Persistent ship-stall detection across checkpoint regressions.
        int shipStallGlobalChunk = -1;
        int shipStallDeaths = 0;
        int shipStallRegressions = 0;

        std::vector<Layer> stack;
        std::vector<uint8_t> committed, pat;
        // Existing per-chunk X data is retained for diagnostics/legacy sidecars.
        std::vector<float> committedX, patX;

        // Full update-by-update successful-path recording and replay.
        std::vector<TraceFrame> committedTrace, patTrace, replayTrace;
        uint64_t physicsUpdate = 0;                    // monotonic training trace sequence
        float currentDt = 0.f;                         // dt for the current postUpdate callback
        float replayIncomingDt = 0.f;                   // game-supplied dt before replay override
        TraceFrame lastReplayExpected{};                 // expected state for last completed replay update
        TraceFrame lastReplayActual{};                    // actual state for last completed replay update
        bool hasLastReplayActual = false;
        float trainingTimeScale = 1.f;                  // scheduler scale requested for the active training session
        float replayRecordedTimeScale = 1.f;             // metadata only; replay scheduler always runs at 1x

        // Hierarchical section training: learn locally with the existing
        // checkpoint search, then validate the entire portal-to-portal section
        // from its start checkpoint in one uninterrupted input stream.
        uint8_t currentModeKey = 0;
        bool modeKeyInitialized = false;
        bool sectionBoundaryPending = false;
        bool finalSectionPending = false;
        bool sectionValidationActive = false;
        bool sectionValidationDeathPending = false;
        bool sectionValidationCompletePending = false;
        bool sectionValidationIsFinal = false;
        bool sectionValidationReachedLevelComplete = false;
        size_t sectionStartChunk = 0;
        size_t sectionStartTrace = 0;
        size_t sectionStartDepth = 0;
        size_t sectionValidationIndex = 0;
        size_t sectionValidationFailureIndex = 0;
        int sectionValidationFailureVdir = 0;
        int sectionValidationWorstUpdate = -1;
        size_t sectionValidationMismatchCount = 0;
        float sectionValidationMaxPositionError = 0.f;
        float sectionValidationMaxVelocityError = 0.f;
        float sectionValidationMaxRotationError = 0.f;
        std::vector<TraceFrame> sectionExpectedFrames;
        std::vector<TraceFrame> sectionActualFrames;
        std::vector<uint8_t> sectionInputPattern;
        std::vector<float> sectionPositionSamples;

        size_t candTraceCount = 0;                     // exact trace prefix at live candidate checkpoint
        size_t replayUpdate = 0;                       // next trace frame to compare
        bool replayTraceLoaded = false;
        bool replayTraceLoadAttempted = false;
        bool replayPrimed = false;                     // replay has reset to the level start
        bool replayDeathStop = false;                   // prevent auto-retry after a replay death
        bool replayWasEnabled = false;                  // detect an off->on toggle to restart replay
        uint32_t physicsStepsSincePostUpdate = 0;       // diagnostic: player physics steps per render callback
        float maxPositionError = 0.f;
        float maxVelocityError = 0.f;
        int divergenceCount = 0;
        int firstDivergence = -1;

        std::vector<float> hazardX;                    // x of every spike, sorted
        std::vector<int> badStarts;                    // absolute chunks where a layer dead-ended
        CheckpointObject* candCp = nullptr;            // checkpoint taken on the live run
        int candG = -1;
        bool candHold = false;
        std::mt19937 rng{std::random_device{}()};

        // per-layer learning state
        std::vector<Ind> archive;
        std::set<std::vector<uint8_t>> tried;
        int curOp = -1, curParentFit = -1, curJ = -1, curW = 0;
        bool curSweep = false;
        std::vector<SweepMove> sweep;
        size_t sweepPos = 0;
        bool sweepBuilt = false;

        // learned across the whole level
        double opW[N_OPS] = {1.0, 1.0, 1.2, 0.5, 0.7, 0.7, 0.4};
        float stepE = 4.f, stepL = 4.f;                // typical size of an "earlier"/"later" move
        float leadMean = 18.f, durMean = 10.f;         // learned lead before the death, hold length

        ~Fields() {
            for (auto& l : stack) if (l.cp) l.cp->release();
            if (candCp) candCp->release();
            CCDirector::get()->getScheduler()->setTimeScale(1.f);
        }
    };

    // ---- helpers ----
    std::filesystem::path savePath() {
        return Mod::get()->getSaveDir() / (std::to_string(m_level->m_levelID.value()) + ".txt");
    }

    std::filesystem::path positionSavePath() {
        auto path = savePath();
        path.replace_extension(".pos");
        return path;
    }

    std::filesystem::path traceSavePath() {
        auto path = savePath();
        path.replace_extension(".trace");
        return path;
    }

    void pressButton(bool down) {
        bool dual = m_gameState.m_isDualMode && m_player2 != nullptr;
        if (down) {
            m_player1->pushButton(PlayerButton::Jump);
            if (dual) m_player2->pushButton(PlayerButton::Jump);
        } else {
            m_player1->releaseButton(PlayerButton::Jump);
            if (dual) m_player2->releaseButton(PlayerButton::Jump);
        }
    }

    // Capture the state immediately after one PlayerObject::update physics
    // step. This hook is called at the same simulation phase during training
    // and replay; `dt` is the actual delta passed to that player update.
    TraceFrame captureTrace(float dt) {
        auto* p = m_player1;
        TraceFrame f;
        f.update = m_fields->physicsUpdate;
        f.input = m_fields->holding ? 1 : 0;
        f.x = p->getPositionX();
        f.y = p->getPositionY();
        f.vx = static_cast<float>(p->getCurrentXVelocity());
        f.vy = static_cast<float>(p->m_yVelocity);
        f.rotation = p->getRotation();
        f.dt = dt;
        f.flags = 0;

        if (p->m_isOnGround) f.flags |= TRACE_GROUND;
        if (p->m_isUpsideDown) f.flags |= TRACE_UPSIDE_DOWN;
        if (p->m_isShip) f.flags |= TRACE_SHIP;
        if (p->m_isBird) f.flags |= TRACE_BIRD;
        if (p->m_isBall) f.flags |= TRACE_BALL;
        if (p->m_isDart) f.flags |= TRACE_DART;
        if (p->m_isRobot) f.flags |= TRACE_ROBOT;
        if (p->m_isSwing) f.flags |= TRACE_SWING;
        return f;
    }

    // Trim committed trace data to the exact checkpoint from which the current
    // layer starts. A chunk-to-update multiplier is not valid here because a
    // reset callback is a control phase, not a physics update.
    void trimCommittedTrace(size_t updates) {
        if (m_fields->committedTrace.size() > updates) {
            m_fields->committedTrace.resize(updates);
        } else if (m_fields->committedTrace.size() < updates) {
            log::error(
                "Autoplayer: trace/checkpoint invariant failed (trace={} checkpoint={}); keeping available trace",
                m_fields->committedTrace.size(), updates
            );
        }
    }

    // A promoted layer keeps the successful input segment after its checkpoint
    // in Layer::tail so its first attempt can reconstruct the already-tested
    // route. That seed is no longer valid after regressing to this checkpoint:
    // using it again would replay the very instructions that caused the stall.
    // Clear only the seed pattern; preserve the checkpoint's captured button
    // state (holdAtCp), committed prefix, banned prefixes, and learned weights.
    void clearRegressionSeed(const char* reason) {
        if (m_fields->stack.empty()) return;

        auto& layer = m_fields->stack.back();
        const size_t oldLength = layer.tail.size();
        layer.tail.clear();

        log::warn(
            "Autoplayer: regression cleared seed pattern reason={} oldChunks={} depth={} checkpoint={} holdAtCp={}",
            reason, oldLength, m_fields->stack.size(), layer.len, layer.holdAtCp
        );
    }

    // Binary trace layout is written field-by-field (no compiler padding).
    bool saveTrace() {
        std::ofstream f(traceSavePath(), std::ios::binary | std::ios::trunc);
        if (!f) return false;

        const char magic[8] = {'G', 'D', 'T', 'R', 'A', 'C', 'E', '1'};
        const uint32_t version = 3;
        const uint32_t searchTicksPerChunk = TICKS;
        const float recordedTimeScale = m_fields->trainingTimeScale;
        const uint64_t count = static_cast<uint64_t>(m_fields->committedTrace.size());

        f.write(magic, sizeof(magic));
        f.write(reinterpret_cast<const char*>(&version), sizeof(version));
        f.write(reinterpret_cast<const char*>(&searchTicksPerChunk), sizeof(searchTicksPerChunk));
        f.write(reinterpret_cast<const char*>(&recordedTimeScale), sizeof(recordedTimeScale));
        f.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (size_t i = 0; i < m_fields->committedTrace.size(); ++i) {
            const auto& t = m_fields->committedTrace[i];
            const uint64_t sequentialUpdate = static_cast<uint64_t>(i);

            f.write(reinterpret_cast<const char*>(&sequentialUpdate), sizeof(sequentialUpdate));
            f.write(reinterpret_cast<const char*>(&t.input), sizeof(t.input));
            f.write(reinterpret_cast<const char*>(&t.x), sizeof(t.x));
            f.write(reinterpret_cast<const char*>(&t.y), sizeof(t.y));
            f.write(reinterpret_cast<const char*>(&t.vx), sizeof(t.vx));
            f.write(reinterpret_cast<const char*>(&t.vy), sizeof(t.vy));
            f.write(reinterpret_cast<const char*>(&t.rotation), sizeof(t.rotation));
            f.write(reinterpret_cast<const char*>(&t.dt), sizeof(t.dt));
            f.write(reinterpret_cast<const char*>(&t.flags), sizeof(t.flags));

            if (!f) return false;
        }

        f.flush();
        return static_cast<bool>(f);
    }

    // The Cocos scheduler time scale is GLOBAL. Keep it enabled only while
    // search is actively running, and always return it to 1x for ordinary play
    // and replay. The existing search algorithm remains unchanged.
    void setSchedulerTimeScale(float requested, const char* reason) {
        float target = requested;
        if (!std::isfinite(target) || target <= 0.f) {
            log::warn(
                "Autoplayer: invalid scheduler scale {} requested by {}; using 1x",
                requested, reason
            );
            target = 1.f;
        }
        target = std::clamp(target, 0.1f, 8.f);

        auto* scheduler = CCDirector::get()->getScheduler();
        const float actual = scheduler->getTimeScale();
        if (!std::isfinite(actual) || std::abs(actual - target) > 0.001f) {
            scheduler->setTimeScale(target);
            log::info(
                "Autoplayer: scheduler scale set reason={} old={:.3f} new={:.3f}",
                reason, actual, target
            );
        }
    }

    void enforceTrainingTimeScale(const char* reason) {
        if (!m_fields->engaged || !m_fields->started || m_fields->done ||
            !Mod::get()->getSettingValue<bool>("run-search")) {
            return;
        }

        setSchedulerTimeScale(m_fields->trainingTimeScale, reason);
    }

    void restoreNormalTimeScale(const char* reason) {
        setSchedulerTimeScale(1.f, reason);
    }

    bool loadTrace() {
        std::ifstream f(traceSavePath(), std::ios::binary);
        if (!f) return false;

        char magic[8]{};
        uint32_t version = 0;
        uint32_t savedTicks = 0;
        float savedTimeScale = 1.f;
        uint64_t count = 0;

        f.read(magic, sizeof(magic));
        f.read(reinterpret_cast<char*>(&version), sizeof(version));
        f.read(reinterpret_cast<char*>(&savedTicks), sizeof(savedTicks));
        f.read(reinterpret_cast<char*>(&savedTimeScale), sizeof(savedTimeScale));
        f.read(reinterpret_cast<char*>(&count), sizeof(count));

        const char expectedMagic[8] = {'G', 'D', 'T', 'R', 'A', 'C', 'E', '1'};
        if (!f || !std::equal(std::begin(magic), std::end(magic), std::begin(expectedMagic)) ||
            version != 3 || savedTicks != static_cast<uint32_t>(TICKS) ||
            !std::isfinite(savedTimeScale) || savedTimeScale <= 0.f || savedTimeScale > 8.f) {
            log::error("Autoplayer: invalid/incompatible physics trace header");
            return false;
        }

        if (count == 0 || count > 100000000ULL ||
            count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            log::error("Autoplayer: invalid physics trace frame count {}", count);
            return false;
        }

        std::vector<TraceFrame> loaded;
        loaded.reserve(static_cast<size_t>(count));

        for (uint64_t i = 0; i < count; ++i) {
            TraceFrame t;
            f.read(reinterpret_cast<char*>(&t.update), sizeof(t.update));
            f.read(reinterpret_cast<char*>(&t.input), sizeof(t.input));
            f.read(reinterpret_cast<char*>(&t.x), sizeof(t.x));
            f.read(reinterpret_cast<char*>(&t.y), sizeof(t.y));
            f.read(reinterpret_cast<char*>(&t.vx), sizeof(t.vx));
            f.read(reinterpret_cast<char*>(&t.vy), sizeof(t.vy));
            f.read(reinterpret_cast<char*>(&t.rotation), sizeof(t.rotation));
            f.read(reinterpret_cast<char*>(&t.dt), sizeof(t.dt));
            f.read(reinterpret_cast<char*>(&t.flags), sizeof(t.flags));

            if (!f || t.update != i || t.input > 1 ||
                !std::isfinite(t.x) || !std::isfinite(t.y) ||
                !std::isfinite(t.vx) || !std::isfinite(t.vy) ||
                !std::isfinite(t.rotation) || !std::isfinite(t.dt)) {
                log::error("Autoplayer: corrupt physics trace at frame {}", i);
                return false;
            }
            loaded.push_back(t);
        }

        m_fields->replayTrace = std::move(loaded);
        m_fields->replayRecordedTimeScale = savedTimeScale;
        m_fields->replayUpdate = 0;
        m_fields->replayTraceLoaded = true;
        m_fields->replayTraceLoadAttempted = true;
        m_fields->replayPrimed = false;
        m_fields->replayFinished = false;
        return true;
    }

    // Apply only the pattern state. Training captures the just-completed frame
    // before this call; the selected state therefore applies to the next update.
    void applyInput(bool want, const char* source, int traceChunk) {
        const bool edge = (want != m_fields->holding);
        if (edge && traceChunk >= 150 && traceChunk <= 600 && m_player1) {
            log::info(
                "Autoplayer: {} {} chunk={} x={:.2f} y={:.2f} vy={:.2f} ground={} upsideDown={} percent={:.2f}",
                source, want ? "PRESS" : "RELEASE", traceChunk,
                m_player1->getPositionX(), m_player1->getPositionY(),
                static_cast<float>(m_player1->m_yVelocity), m_player1->m_isOnGround,
                m_player1->m_isUpsideDown, this->getCurrentPercent()
            );
        }

        if (want && !m_fields->holding) {
            if (m_fields->engaged && m_fields->pressLogs++ < 15) {
                int globalChunk = m_fields->stack.empty()
                    ? m_fields->chunk
                    : static_cast<int>(m_fields->stack.back().len) + m_fields->chunk;
                log::info("Autoplayer: PRESS at local chunk {} (global {}, depth {}, try {})",
                          m_fields->chunk, globalChunk, m_fields->stack.size(), m_fields->tries);
            }
            pressButton(true);
            m_fields->holding = true;
        } else if (!want && m_fields->holding) {
            pressButton(false);
            m_fields->holding = false;
        }
    }

    void cacheHazards() {
        m_fields->hazardX.clear();
        for (auto* obj : CCArrayExt<GameObject*>(m_objects)) {
            if (obj && obj->m_objectType == GameObjectType::Hazard) {
                m_fields->hazardX.push_back(obj->getPositionX());
            }
        }
        std::sort(m_fields->hazardX.begin(), m_fields->hazardX.end());
        log::info("Autoplayer: cached {} hazards", m_fields->hazardX.size());
    }

    bool runwayClear() {
        float px = m_player1->getPositionX();
        auto& hx = m_fields->hazardX;
        auto it = std::lower_bound(hx.begin(), hx.end(), px - RUNWAY_BEHIND);
        return it == hx.end() || *it > px + RUNWAY_AHEAD;
    }

    bool startBanned(int g) {
        for (int b : m_fields->badStarts) if (std::abs(b - g) <= BAN_RADIUS) return true;
        return false;
    }

    bool safeToCheckpoint(int g) {
        if (startBanned(g)) return false;
        if (m_fields->fly) return true;
        return m_player1->m_isOnGround && !m_fields->holding && runwayClear();
    }

    // For ship, put the next checkpoint about SHIP_CHECKPOINT_LEAD chunks
    // before the last known death. If that much room is no longer available,
    // wait until the bot has cleared the known death instead of checkpointing
    // right beside it.
    int shipCheckpointTarget(const Layer& layer) const {
        if (layer.lastDeathChunk < 0) return COMMIT;
        if (layer.lastDeathChunk >= SHIP_CHECKPOINT_LEAD + COMMIT)
            return std::max(COMMIT, layer.lastDeathChunk - SHIP_CHECKPOINT_LEAD);
        return layer.lastDeathChunk + SHIP_CHECKPOINT_CLEAR;
    }

    void clearCand() {
        if (m_fields->candCp) m_fields->candCp->release();
        m_fields->candCp = nullptr;
        m_fields->candG = -1;
        m_fields->candTraceCount = 0;
    }

    void updateFly() {
        auto* p = m_player1;
        if (!p) {
            m_fields->ship = false;
            m_fields->fly = false;
            return;
        }

        m_fields->ship = p->m_isShip;
        m_fields->wave = p->m_isDart;
        m_fields->fly =
            p->m_isShip ||
            p->m_isBird ||
            p->m_isDart ||
            p->m_isSwing;
    }

    // A section boundary is a gameplay-physics transition, not just a visual
    // portal. Include gravity inversion because it changes the control model.
    // Bit layout: ship, bird, ball, dart, robot, spider, swing, upside-down.
    uint8_t modeSignature(const PlayerObject* p) const {
        if (!p) return 0;
        uint8_t key = 0;
        if (p->m_isShip)       key |= 1u << 0;
        if (p->m_isBird)       key |= 1u << 1;
        if (p->m_isBall)       key |= 1u << 2;
        if (p->m_isDart)       key |= 1u << 3;
        if (p->m_isRobot)      key |= 1u << 4;
        if (p->m_isSpider)     key |= 1u << 5;
        if (p->m_isSwing)      key |= 1u << 6;
        if (p->m_isUpsideDown) key |= 1u << 7;
        return key;
    }

    // A very short release between two cube jumps may be bridged, but only
    // for an isolated pair. Never chain several jumps into one long hold:
    // in cube mode, holding through a landing can trigger another unwanted jump.
    // Other gamemodes are deliberately excluded.
    bool isCubeMode() const {
        auto* p = m_player1;
        return p &&
            !p->m_isShip && !p->m_isBird && !p->m_isBall &&
            !p->m_isDart && !p->m_isRobot && !p->m_isSpider &&
            !p->m_isSwing;
    }

    void mergeRapidCubeJumps(std::vector<uint8_t>& pattern) {
        if (!isCubeMode() || pattern.size() < 3) return;

        struct Run {
            size_t start;
            size_t end; // exclusive
            bool pressed;
        };

        std::vector<Run> runs;
        size_t start = 0;
        bool state = pattern[0] != 0;
        for (size_t i = 1; i <= pattern.size(); ++i) {
            if (i == pattern.size() || (pattern[i] != 0) != state) {
                runs.push_back({start, i, state});
                if (i < pattern.size()) {
                    start = i;
                    state = pattern[i] != 0;
                }
            }
        }

        // Bridge only a one-chunk release between two presses, and only when
        // neither neighbouring gap is also eligible. This prevents a sequence
        // of rapid taps from becoming one extended hold.
        for (size_t i = 1; i + 1 < runs.size(); ++i) {
            if (runs[i].pressed || !runs[i - 1].pressed || !runs[i + 1].pressed)
                continue;

            const size_t gap = runs[i].end - runs[i].start;
            if (gap == 0 || gap > static_cast<size_t>(CUBE_RAPID_GAP))
                continue;

            const bool previousGapAlsoShort =
                i >= 3 && !runs[i - 2].pressed &&
                (runs[i - 2].end - runs[i - 2].start) <= static_cast<size_t>(CUBE_RAPID_GAP);
            const bool nextGapAlsoShort =
                i + 2 < runs.size() && !runs[i + 2].pressed &&
                (runs[i + 2].end - runs[i + 2].start) <= static_cast<size_t>(CUBE_RAPID_GAP);

            if (previousGapAlsoShort || nextGapAlsoShort)
                continue;

            std::fill(pattern.begin() + static_cast<long>(runs[i].start),
                      pattern.begin() + static_cast<long>(runs[i].end), 1);
        }
    }

    // forget the current layer's attempts (the learned weights/steps are kept)
    void resetLayerState() {
        m_fields->archive.clear();
        m_fields->tried.clear();
        m_fields->curOp = -1;
        m_fields->curParentFit = -1;
        m_fields->curSweep = false;
        m_fields->sweep.clear();
        m_fields->sweepPos = 0;
        m_fields->sweepBuilt = false;
    }

    void clearSectionValidationBuffers() {
        m_fields->sectionExpectedFrames.clear();
        m_fields->sectionActualFrames.clear();
        m_fields->sectionInputPattern.clear();
        m_fields->sectionPositionSamples.clear();
        m_fields->sectionValidationIndex = 0;
        m_fields->sectionValidationFailureIndex = 0;
        m_fields->sectionValidationFailureVdir = 0;
        m_fields->sectionValidationWorstUpdate = -1;
        m_fields->sectionValidationMismatchCount = 0;
        m_fields->sectionValidationMaxPositionError = 0.f;
        m_fields->sectionValidationMaxVelocityError = 0.f;
        m_fields->sectionValidationMaxRotationError = 0.f;
        m_fields->sectionValidationDeathPending = false;
        m_fields->sectionValidationCompletePending = false;
        m_fields->sectionValidationReachedLevelComplete = false;
        m_fields->sectionValidationIsFinal = false;
    }

    // Freeze the locally learned route for the current section, rewind to the
    // section's start checkpoint, and play its recorded physics-step inputs
    // continuously. No mutation, checkpoint promotion, or reset occurs during
    // this validation pass.
    bool beginSectionValidation(bool finalSection) {
        if (m_fields->stack.empty()) {
            log::error("Autoplayer: cannot validate section without a start layer");
            return false;
        }
        if (m_fields->sectionStartDepth >= m_fields->stack.size()) {
            log::error(
                "Autoplayer: section start depth is invalid (startDepth={} stack={})",
                m_fields->sectionStartDepth, m_fields->stack.size()
            );
            return false;
        }

        const size_t localChunkRequested =
            static_cast<size_t>(std::max(0, m_fields->chunk)) +
            (finalSection ? size_t{1} : size_t{0});
        const size_t localChunkCount = std::min(
            localChunkRequested, m_fields->pat.size()
        );
        const size_t globalEndChunk = m_fields->stack.back().len + localChunkCount;

        std::vector<uint8_t> allInputs = m_fields->committed;
        if (localChunkCount > 0) {
            allInputs.insert(
                allInputs.end(),
                m_fields->pat.begin(),
                m_fields->pat.begin() + static_cast<long>(localChunkCount)
            );
        }

        std::vector<float> allX = m_fields->committedX;
        if (m_fields->patX.size() < localChunkCount) {
            m_fields->patX.resize(
                localChunkCount,
                m_player1 ? m_player1->getPositionX() : 0.f
            );
        }
        if (localChunkCount > 0) {
            allX.insert(
                allX.end(),
                m_fields->patX.begin(),
                m_fields->patX.begin() + static_cast<long>(localChunkCount)
            );
        }

        if (allX.size() < globalEndChunk) {
            allX.resize(
                globalEndChunk,
                m_player1 ? m_player1->getPositionX() : 0.f
            );
        }

        std::vector<TraceFrame> allTrace = m_fields->committedTrace;
        allTrace.insert(
            allTrace.end(),
            m_fields->patTrace.begin(),
            m_fields->patTrace.end()
        );

        if (m_fields->sectionStartChunk > globalEndChunk ||
            m_fields->sectionStartChunk > allInputs.size() ||
            globalEndChunk > allInputs.size() ||
            m_fields->sectionStartChunk > allX.size() ||
            globalEndChunk > allX.size() ||
            m_fields->sectionStartTrace > allTrace.size()) {
            log::error(
                "Autoplayer: section capture invariant failed startChunk={} endChunk={} inputCount={} startTrace={} traceCount={}",
                m_fields->sectionStartChunk, globalEndChunk, allInputs.size(),
                m_fields->sectionStartTrace, allTrace.size()
            );
            return false;
        }

        m_fields->sectionInputPattern.assign(
            allInputs.begin() + static_cast<long>(m_fields->sectionStartChunk),
            allInputs.begin() + static_cast<long>(globalEndChunk)
        );
        m_fields->sectionPositionSamples.assign(
            allX.begin() + static_cast<long>(m_fields->sectionStartChunk),
            allX.begin() + static_cast<long>(std::min(globalEndChunk, allX.size()))
        );
        m_fields->sectionExpectedFrames.assign(
            allTrace.begin() + static_cast<long>(m_fields->sectionStartTrace),
            allTrace.end()
        );

        if (m_fields->sectionExpectedFrames.empty()) {
            log::warn("Autoplayer: skipping empty section validation");
            return false;
        }

        const auto& savedRoot = m_fields->stack[m_fields->sectionStartDepth];
        if (savedRoot.len != m_fields->sectionStartChunk ||
            savedRoot.traceLen != m_fields->sectionStartTrace) {
            log::error(
                "Autoplayer: section root does not match saved boundary (layerChunk={} expectedChunk={} layerTrace={} expectedTrace={})",
                savedRoot.len, m_fields->sectionStartChunk,
                savedRoot.traceLen, m_fields->sectionStartTrace
            );
            return false;
        }

        // A boundary checkpoint is the root of this section. Remove only the
        // provisional micro-checkpoints created inside this section.
        clearCand();
        while (m_fields->stack.size() > m_fields->sectionStartDepth + 1) {
            Layer discarded = std::move(m_fields->stack.back());
            m_fields->stack.pop_back();
            if (discarded.cp) discarded.cp->release();
        }

        auto& root = m_fields->stack.back();

        m_fields->committed.resize(m_fields->sectionStartChunk);
        m_fields->committedX.resize(m_fields->sectionStartChunk);
        trimCommittedTrace(m_fields->sectionStartTrace);
        root.tail = m_fields->sectionInputPattern;

        m_fields->pat.clear();
        m_fields->patX.clear();
        m_fields->patTrace.clear();
        m_fields->chunk = 0;
        m_fields->tickIn = 0;
        m_fields->tries = 0;
        resetLayerState();

        m_fields->sectionValidationIsFinal = finalSection;
        m_fields->sectionValidationActive = true;
        m_fields->sectionValidationDeathPending = false;
        m_fields->sectionValidationCompletePending = false;
        m_fields->sectionValidationReachedLevelComplete = false;
        m_fields->sectionValidationIndex = 0;
        m_fields->sectionValidationFailureIndex = 0;
        m_fields->sectionValidationFailureVdir = 0;
        m_fields->sectionValidationWorstUpdate = -1;
        m_fields->sectionValidationMismatchCount = 0;
        m_fields->sectionValidationMaxPositionError = 0.f;
        m_fields->sectionValidationMaxVelocityError = 0.f;
        m_fields->sectionValidationMaxRotationError = 0.f;
        m_fields->sectionActualFrames.clear();
        m_fields->sectionBoundaryPending = false;
        m_fields->finalSectionPending = false;
        m_fields->dead = false;
        m_fields->needRewind = true;

        log::info(
            "Autoplayer: SECTION VALIDATION START chunks={} physicsUpdates={} startChunk={} modeKey=0x{:02X} final={}",
            m_fields->sectionInputPattern.size(), m_fields->sectionExpectedFrames.size(),
            m_fields->sectionStartChunk, m_fields->currentModeKey, finalSection
        );

        // Rewind immediately to the section root. Unlike a normal search rewind,
        // do not call newTrial(): validation consumes the saved TraceFrame inputs.
        rewind();
        m_fields->sectionValidationActive = true;
        return true;
    }

    void resumeSectionTrainingAfterValidationFailure(const char* reason) {
        if (m_fields->stack.empty()) return;

        const size_t failIndex = std::min(
            m_fields->sectionValidationFailureIndex,
            m_fields->sectionExpectedFrames.empty()
                ? size_t{0}
                : m_fields->sectionExpectedFrames.size() - 1
        );
        const size_t frameCount = std::max<size_t>(1, m_fields->sectionExpectedFrames.size());
        const size_t chunkCount = m_fields->sectionInputPattern.size();
        const int failChunk = chunkCount == 0
            ? 0
            : std::min(
                static_cast<int>(chunkCount - 1),
                static_cast<int>((failIndex * chunkCount) / frameCount)
            );
        const int failVdir = m_fields->sectionValidationFailureVdir;
        const auto seed = m_fields->sectionInputPattern;

        auto& root = m_fields->stack.back();
        root.tail = seed;

        m_fields->sectionValidationActive = false;
        m_fields->sectionBoundaryPending = false;
        m_fields->finalSectionPending = false;
        m_fields->sectionValidationDeathPending = false;
        m_fields->sectionValidationCompletePending = false;
        m_fields->sectionValidationReachedLevelComplete = false;
        m_fields->dead = false;
        m_fields->needRewind = true;
        m_fields->pat.clear();
        m_fields->patX.clear();
        m_fields->patTrace.clear();
        m_fields->chunk = 0;
        m_fields->tickIn = 0;
        m_fields->tries = 1;

        resetLayerState();
        if (!seed.empty()) {
            archiveAdd(seed, failChunk, failVdir);
            m_fields->tried.insert(seed);
        }

        log::warn(
            "Autoplayer: SECTION VALIDATION FAILED reason={} failureFrame={} approxChunk={} maxPosition={:.3f} maxVelocity={:.3f} maxRotation={:.3f}; returning to local search at chunk {}",
            reason, failIndex, failChunk,
            m_fields->sectionValidationMaxPositionError,
            m_fields->sectionValidationMaxVelocityError,
            m_fields->sectionValidationMaxRotationError,
            m_fields->sectionStartChunk
        );

        clearSectionValidationBuffers();
        updateFly();
        m_fields->currentModeKey = modeSignature(m_player1);
        m_fields->modeKeyInitialized = true;
    }

    void saveValidatedLevelAndFinish() {
        std::string s;
        for (auto v : m_fields->committed) s += v ? '1' : '0';
        std::ofstream(savePath()) << s;

        bool positionsSaved = (m_fields->committedX.size() == m_fields->committed.size());
        if (positionsSaved) {
            std::ofstream pf(positionSavePath());
            pf << std::fixed << std::setprecision(4);
            for (float x : m_fields->committedX) pf << x << '\n';
            positionsSaved = static_cast<bool>(pf);
        }

        const bool traceSaved = saveTrace();
        m_fields->replayTraceLoaded = false;
        m_fields->replayTraceLoadAttempted = false;
        m_fields->replayTrace.clear();
        m_fields->done = true;
        m_fields->engaged = false;
        m_fields->sectionValidationActive = false;
        restoreNormalTimeScale("training completed");

        log::info(
            "Autoplayer: LEVEL BEATEN after section validation; saved {} chunks and {} player physics updates (input={}, world-X={}, trace={})",
            m_fields->committed.size(), m_fields->committedTrace.size(),
            savePath().string(), positionsSaved, traceSaved
        );
        Notification::create(
            traceSaved ? "Level beaten - validated physics trace saved" : "Level beaten - trace save failed",
            traceSaved ? NotificationIcon::Success : NotificationIcon::Error
        )->show();
    }

    void finishSectionValidation() {
        if (!m_fields->sectionValidationActive || m_fields->stack.empty()) return;

        const bool finalSection =
            m_fields->sectionValidationIsFinal ||
            m_fields->sectionValidationReachedLevelComplete;
        const bool completeTrace =
            m_fields->sectionActualFrames.size() >=
            m_fields->sectionExpectedFrames.size();
        const bool qualityPassed =
            m_fields->sectionValidationMaxPositionError <= SECTION_MAX_POSITION_ERROR &&
            m_fields->sectionValidationMaxVelocityError <= SECTION_MAX_VELOCITY_ERROR &&
            m_fields->sectionValidationMaxRotationError <= SECTION_MAX_ROTATION_ERROR &&
            m_fields->sectionValidationMismatchCount == 0;

        if (!completeTrace || !qualityPassed ||
            (finalSection && !m_fields->sectionValidationReachedLevelComplete)) {
            if (m_fields->sectionValidationWorstUpdate >= 0) {
                m_fields->sectionValidationFailureIndex = static_cast<size_t>(
                    m_fields->sectionValidationWorstUpdate
                );
            } else if (!m_fields->sectionExpectedFrames.empty()) {
                m_fields->sectionValidationFailureIndex =
                    m_fields->sectionExpectedFrames.size() - 1;
            }
            resumeSectionTrainingAfterValidationFailure(
                !completeTrace ? "trace ended early" :
                (!qualityPassed ? "trajectory drift" : "level completion was not reproduced")
            );
            return;
        }

        // Commit the fresh trace from the continuous pass, not the provisional
        // collection of checkpoint-reset training traces.
        const size_t expectedChunkEnd =
            m_fields->sectionStartChunk + m_fields->sectionInputPattern.size();
        m_fields->committed.resize(m_fields->sectionStartChunk);
        m_fields->committed.insert(
            m_fields->committed.end(),
            m_fields->sectionInputPattern.begin(),
            m_fields->sectionInputPattern.end()
        );

        m_fields->committedX.resize(m_fields->sectionStartChunk);
        m_fields->committedX.insert(
            m_fields->committedX.end(),
            m_fields->sectionPositionSamples.begin(),
            m_fields->sectionPositionSamples.end()
        );
        if (m_fields->committedX.size() < expectedChunkEnd) {
            m_fields->committedX.resize(
                expectedChunkEnd,
                m_player1 ? m_player1->getPositionX() : 0.f
            );
        }

        m_fields->committedTrace.resize(m_fields->sectionStartTrace);
        m_fields->committedTrace.insert(
            m_fields->committedTrace.end(),
            m_fields->sectionActualFrames.begin(),
            m_fields->sectionActualFrames.end()
        );

        log::info(
            "Autoplayer: SECTION VALIDATION PASSED chunks={} physicsUpdates={} maxPosition={:.3f} maxVelocity={:.3f} maxRotation={:.3f} final={}",
            m_fields->sectionInputPattern.size(),
            m_fields->sectionActualFrames.size(),
            m_fields->sectionValidationMaxPositionError,
            m_fields->sectionValidationMaxVelocityError,
            m_fields->sectionValidationMaxRotationError,
            finalSection
        );

        m_fields->sectionValidationActive = false;
        m_fields->sectionValidationCompletePending = false;
        m_fields->sectionValidationDeathPending = false;
        m_fields->sectionBoundaryPending = false;
        m_fields->finalSectionPending = false;
        m_fields->dead = false;

        if (finalSection) {
            clearSectionValidationBuffers();
            saveValidatedLevelAndFinish();
            PlayLayer::levelComplete();
            return;
        }

        auto* endpoint = this->createCheckpoint();
        if (!endpoint) {
            m_fields->sectionValidationFailureIndex =
                m_fields->sectionExpectedFrames.empty()
                    ? 0
                    : m_fields->sectionExpectedFrames.size() - 1;
            m_fields->sectionValidationFailureVdir = 0;
            m_fields->sectionValidationActive = true;
            resumeSectionTrainingAfterValidationFailure("could not create section-end checkpoint");
            return;
        }
        endpoint->retain();

        // The old section-root layer remains as a rollback boundary. Its local
        // seed is no longer needed after the full section has passed.
        m_fields->stack.back().tail.clear();
        Layer next;
        next.cp = endpoint;
        next.len = m_fields->committed.size();
        next.traceLen = m_fields->committedTrace.size();
        next.holdAtCp = m_fields->holding;
        next.tail.push_back(m_fields->holding ? 1 : 0);
        m_fields->stack.push_back(std::move(next));

        m_fields->sectionStartChunk = m_fields->committed.size();
        m_fields->sectionStartTrace = m_fields->committedTrace.size();
        m_fields->sectionStartDepth = m_fields->stack.size() - 1;
        updateFly();
        m_fields->currentModeKey = modeSignature(m_player1);
        m_fields->modeKeyInitialized = true;

        m_fields->pat = basePattern(m_fields->stack.back().tail);
        m_fields->patX.clear();
        m_fields->patTrace.clear();
        m_fields->chunk = 0;
        m_fields->tickIn = 0;
        m_fields->tries = 0;
        m_fields->needRewind = false;
        resetLayerState();
        clearSectionValidationBuffers();

        log::info(
            "Autoplayer: advancing to next section at chunk {} traceUpdate={} depth={}",
            m_fields->sectionStartChunk, m_fields->sectionStartTrace,
            m_fields->stack.size()
        );
    }

    // the seed pattern, continued with its last state
    std::vector<uint8_t> basePattern(const std::vector<uint8_t>& tail) {
        std::vector<uint8_t> p(std::max(static_cast<size_t>(HORIZON), tail.size()), 0);
        uint8_t fill = tail.empty() ? 0 : tail.back();
        for (size_t i = 0; i < p.size(); ++i) p[i] = i < tail.size() ? tail[i] : fill;
        return p;
    }

    // ---- learning ----
    void archiveAdd(const std::vector<uint8_t>& pat, int fit, int vd) {
        for (const auto& a : m_fields->archive) if (a.pat == pat) return;
        Ind ind;
        ind.pat = pat;
        ind.fit = fit;
        ind.vdir = vd;
        if (!m_fields->patTrace.empty()) {
            ind.deathY = m_fields->patTrace.back().y;
            ind.deathVy = m_fields->patTrace.back().vy;
        }

        auto candidates = m_fields->archive;
        candidates.push_back(std::move(ind));
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const Ind& a, const Ind& b) { return a.fit > b.fit; });

        // Always retain the best attempt, then preserve genuinely different
        // input trajectories instead of allowing six near-identical variants
        // to crowd every alternative out of the archive.
        std::vector<Ind> selected;
        selected.reserve(ARCHIVE_SIZE);
        if (!candidates.empty()) selected.push_back(candidates.front());
        const int bestFit = candidates.empty() ? 0 : candidates.front().fit;
        auto distance = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
            const size_t n = std::max(a.size(), b.size());
            if (!n) return 0.f;
            size_t diff = a.size() > b.size() ? a.size() - b.size() : b.size() - a.size();
            const size_t common = std::min(a.size(), b.size());
            for (size_t i = 0; i < common; ++i) if (a[i] != b[i]) ++diff;
            return static_cast<float>(diff) / static_cast<float>(n);
        };

        for (size_t i = 1; i < candidates.size() && selected.size() < ARCHIVE_SIZE; ++i) {
            float nearest = 1.f;
            for (const auto& kept : selected)
                nearest = std::min(nearest, distance(candidates[i].pat, kept.pat));
            const bool nearBest = candidates[i].fit >= bestFit - 2;
            const bool distinct = nearest >= (m_fields->fly ? 0.035f : 0.08f);
            if (nearBest || distinct) selected.push_back(candidates[i]);
        }
        // Fill any remaining slots with the next-best candidates.
        for (size_t i = 1; i < candidates.size() && selected.size() < ARCHIVE_SIZE; ++i) {
            bool exists = false;
            for (const auto& kept : selected) if (kept.pat == candidates[i].pat) { exists = true; break; }
            if (!exists) selected.push_back(candidates[i]);
        }
        m_fields->archive = std::move(selected);
    }

    // reward/punish the operator (and step size) that produced the attempt that just finished
    void credit(int fit) {
        if (m_fields->curOp < 0 || m_fields->curParentFit < 0) return;
        bool win = fit > m_fields->curParentFit + 1;
        int op = m_fields->curOp;
        double& w = m_fields->opW[op];
        w = win ? std::min(w * 1.4, 10.0) : std::max(w * 0.98, 0.15);
        if (op == OP_EARLY || op == OP_SHRINK) {
            m_fields->stepE = win ? std::min(m_fields->stepE * 1.4f, 24.f) : std::max(m_fields->stepE * 0.93f, 1.f);
        }
        if (op == OP_LATE || op == OP_EXTEND) {
            m_fields->stepL = win ? std::min(m_fields->stepL * 1.4f, 24.f) : std::max(m_fields->stepL * 0.93f, 1.f);
        }
        if (op == OP_ADD && win && m_fields->curJ >= 0) {
            m_fields->leadMean = 0.8f * m_fields->leadMean + 0.2f * static_cast<float>(m_fields->curParentFit - m_fields->curJ);
            m_fields->durMean = 0.8f * m_fields->durMean + 0.2f * static_cast<float>(m_fields->curW);
        }
        m_fields->curOp = -1;
    }

    // Every edge of the best attempt (before its death), shifted earlier/later by a range of amounts,
    // alone and as a whole tap. Small moves first, then edges nearest the death first.
    void buildSweep() {
        m_fields->sweep.clear();
        m_fields->sweepPos = 0;
        m_fields->sweepBuilt = true;
        if (m_fields->archive.empty()) return;
        const Ind& best = m_fields->archive[0];
        std::vector<int> e = toEdges(best.pat);
        const int d = best.fit;
        std::vector<SweepMove> mv;
        for (size_t i = 0; i < e.size(); ++i) {
            if (e[i] > d) continue;   // happens after the death, can't matter
            for (int k = 0; k < N_SWEEP_DELTAS; ++k) {
                for (int sgn : {-1, 1}) {
                    int delta = sgn * SWEEP_DELTAS[k];
                    mv.push_back({static_cast<int>(i), delta, false});
                    if (i % 2 == 0 && i + 1 < e.size()) mv.push_back({static_cast<int>(i), delta, true});
                }
            }
        }
        std::stable_sort(mv.begin(), mv.end(), [&](const SweepMove& a, const SweepMove& b) {
            int da = std::abs(a.delta), db = std::abs(b.delta);
            if (da != db) return da < db;
            return std::abs(d - e[static_cast<size_t>(a.edge)]) < std::abs(d - e[static_cast<size_t>(b.edge)]);
        });
        if (mv.size() > SWEEP_MAX) mv.resize(SWEEP_MAX);
        log::info("Autoplayer: stuck at {} chunks, starting a focused edge sweep of {} moves over {} edges",
                  d, mv.size(), e.size());
        m_fields->sweep = std::move(mv);
    }

    std::vector<uint8_t> applySweep(const Ind& par, const SweepMove& mv) {
        std::vector<uint8_t> p = par.pat;
        std::vector<int> e = toEdges(p);
        size_t k = static_cast<size_t>(mv.edge);
        if (k >= e.size()) return p;
        const int len = static_cast<int>(p.size());
        int lo = k > 0 ? e[k - 1] + 1 : 0;
        if (mv.tap && k + 1 < e.size()) {
            int hi = (k + 2 < e.size()) ? e[k + 2] - 1 : len - 1;
            int span = e[k + 1] - e[k];
            if (hi - span < lo) return p;
            int ns = std::min(hi - span, std::max(lo, e[k] + mv.delta));
            e[k] = ns;
            e[k + 1] = ns + span;
        } else {
            int hi = (k + 1 < e.size()) ? e[k + 1] - 1 : len - 1;
            if (hi < lo) return p;
            e[k] = std::min(hi, std::max(lo, e[k] + mv.delta));
        }
        return fromEdges(e, p.size());
    }

    int pickOp(const Ind& par) {
        double w[N_OPS];
        for (int i = 0; i < N_OPS; ++i) w[i] = m_fields->opW[i];
        if (toEdges(par.pat).empty()) {
            for (int i = 0; i < N_OPS; ++i) w[i] = 0.0;
            w[OP_ADD] = m_fields->opW[OP_ADD] * 2.0;
            w[OP_RANDOM] = m_fields->opW[OP_RANDOM];
        } else if (par.vdir > 0) {
            w[OP_EARLY] *= 3.0;      // died while rising: pressed too late
        } else if (par.vdir < 0) {
            w[OP_LATE] *= 3.0;       // died while falling: pressed too early
        } else {
            w[OP_ADD] *= 2.0;
        }
        if (m_fields->ship || m_fields->wave) {
            w[OP_RANDOM] *= 1.25; // prefer targeted edge/pulse experiments over wholesale noise
            w[OP_ADD] *= m_fields->ship ? 1.8 : 1.4;
        } else if (m_fields->fly) {
            w[OP_RANDOM] *= 2.0;
        }
        std::discrete_distribution<int> dist(w, w + N_OPS);
        return dist(m_fields->rng);
    }

    // Flight modes benefit from changing a complete press pulse as a unit.
    // Moving only one edge can accidentally change both its timing and width,
    // which is often the wrong experiment for ship/wave control.
    std::vector<uint8_t> mutateFlightPulse(const Ind& par) {
        std::vector<uint8_t> p = par.pat;
        std::vector<int> edges = toEdges(p);
        const int death = std::max(0, par.fit);
        std::vector<size_t> pulses;
        for (size_t i = 0; i + 1 < edges.size(); i += 2) {
            if (edges[i] <= death && edges[i + 1] > edges[i]) pulses.push_back(i);
        }
        auto& rng = m_fields->rng;
        std::uniform_int_distribution<int> deltaDist(-8, 8);
        std::uniform_int_distribution<int> widthDist(1, 8);
        std::uniform_int_distribution<int> opDist(0, 3);
        const int op = opDist(rng);

        if (pulses.empty()) {
            // With no learned pulse, add a short exploratory correction before
            // the death point; ship corrections are deliberately earlier.
            const int leadMin = m_fields->ship ? 24 : 4;
            const int leadMax = m_fields->ship ? 96 : 32;
            const int lead = leadMin + static_cast<int>(rng() % (leadMax - leadMin + 1));
            const int begin = std::max(0, death - lead);
            const int width = m_fields->ship ? 2 + static_cast<int>(rng() % 8)
                                             : 1 + static_cast<int>(rng() % 4);
            if (p.size() < static_cast<size_t>(begin + width))
                p.resize(static_cast<size_t>(begin + width), p.empty() ? 0 : p.back());
            for (int c = begin; c < begin + width; ++c) p[static_cast<size_t>(c)] ^= 1;
            return p;
        }

        // Prefer pulses closest to the current failure, but retain some
        // exploration of earlier corrections (especially important for ship).
        size_t pulseIndex = pulses.back();
        if (pulses.size() > 1 && (rng() % 4) == 0)
            pulseIndex = pulses[static_cast<size_t>(rng() % pulses.size())];
        const size_t pulseNumber = static_cast<size_t>(std::distance(pulses.begin(),
            std::find(pulses.begin(), pulses.end(), pulseIndex)));
        (void)pulseNumber;
        const int start = edges[pulseIndex];
        const int finish = edges[pulseIndex + 1];
        const int len = static_cast<int>(p.size());

        if (op == 0) { // slide the entire pulse earlier/later, preserving width
            int delta = deltaDist(rng);
            if (delta == 0) delta = 1;
            int lo = pulseIndex > 0 ? edges[pulseIndex - 1] + 1 : 0;
            int hi = pulseIndex + 2 < edges.size() ? edges[pulseIndex + 2] - 1 : len - 1;
            int ns = std::min(hi - (finish - start), std::max(lo, start + delta));
            if (ns >= lo && ns + (finish - start) <= hi) {
                edges[pulseIndex] = ns;
                edges[pulseIndex + 1] = ns + (finish - start);
            }
        } else if (op == 1) { // change pulse duration without moving its start
            int delta = widthDist(rng);
            // Use the death trajectory as a weak directional hint. If the
            // player dies while rising, test releasing sooner; while falling,
            // test holding longer. Keep some random exploration in either case.
            if ((m_fields->ship || m_fields->wave) && par.vdir != 0 && (rng() % 4) != 0)
                delta = par.vdir > 0 ? -delta : delta;
            else if (rng() & 1)
                delta = -delta;
            const float speedScale = std::clamp(std::abs(par.deathVy) / 8.f, 0.75f, 1.75f);
            delta = std::max(1, static_cast<int>(std::lround(std::abs(delta) * speedScale))) *
                    (delta < 0 ? -1 : 1);
            int hi = pulseIndex + 2 < edges.size() ? edges[pulseIndex + 2] - 1 : len - 1;
            edges[pulseIndex + 1] = std::min(hi, std::max(start + 1, finish + delta));
        } else if (op == 2) { // split a long hold into two shorter corrections
            if (finish - start >= 4) {
                int split = start + (finish - start) / 2;
                edges.insert(edges.begin() + static_cast<long>(pulseIndex + 1), split);
                edges.insert(edges.begin() + static_cast<long>(pulseIndex + 2), split + 1);
            }
        } else { // merge two nearby pulses by removing the release/next press
            if (pulseIndex + 3 < edges.size() &&
                edges[pulseIndex + 2] - edges[pulseIndex + 1] <= (m_fields->wave ? 12 : 20)) {
                edges.erase(edges.begin() + static_cast<long>(pulseIndex + 1),
                            edges.begin() + static_cast<long>(pulseIndex + 3));
            }
        }
        return fromEdges(edges, p.size());
    }

    std::vector<uint8_t> mutate(const Ind& par, int op) {
        std::vector<uint8_t> p = par.pat;
        const int len = static_cast<int>(p.size());
        std::vector<int> e = toEdges(p);
        const int d = par.fit;
        auto& rng = m_fields->rng;
        std::uniform_real_distribution<float> U(0.f, 1.f);
        std::normal_distribution<float> N(0.f, 1.f);

        if ((m_fields->ship || m_fields->wave) && U(rng) < (m_fields->ship ? 0.62f : 0.72f))
            return mutateFlightPulse(par);

        std::vector<int> before;   // edges that happen before the death
        for (size_t i = 0; i < e.size(); ++i) if (e[i] <= d) before.push_back(static_cast<int>(i));

        // parity: -1 any edge, 0 press edges, 1 release edges. Prefer the edges nearest the death.
        auto pickEdge = [&](int parity) -> int {
            std::vector<int> c;
            for (int i : before) if (parity < 0 || (i % 2) == parity) c.push_back(i);
            if (c.empty()) {
                for (size_t i = 0; i < e.size(); ++i) {
                    if (parity < 0 || (static_cast<int>(i) % 2) == parity) c.push_back(static_cast<int>(i));
                }
            }
            if (c.empty()) return -1;
            float r = U(rng);
            if (r < 0.5f) return c.back();
            if (r < 0.75f && c.size() >= 2) return c[c.size() - 2];
            return c[static_cast<size_t>(rng() % c.size())];
        };
        auto stepOf = [&](bool early) -> int {
            if (U(rng) < BIG_MOVE) return 24 + static_cast<int>(U(rng) * 40.f);   // occasional big step
            float s = early ? m_fields->stepE : m_fields->stepL;
            return std::max(1, static_cast<int>(std::lround(s * (0.3f + 1.2f * U(rng)))));
        };
        auto shiftEdge = [&](int i, int delta) -> bool {
            size_t k = static_cast<size_t>(i);
            int lo = k > 0 ? e[k - 1] + 1 : 0;
            int hi = (k + 1 < e.size()) ? e[k + 1] - 1 : len - 1;
            if (hi < lo) hi = lo;
            int ne = std::min(hi, std::max(lo, e[k] + delta));
            if (ne == e[k]) return false;
            e[k] = ne;
            return true;
        };
        auto addTap = [&]() -> std::vector<uint8_t> {
            int latest = std::max(0, d - 1);
            int j = 0;

            // With no existing button presses, explore the entire available
            // timing window instead of always starting at the learned lead time.
            if (e.empty()) {
                std::uniform_int_distribution<int> startDist(0, latest);
                j = startDist(rng);
            } else {
                // Ship may need several altitude corrections before the obstacle.
                // Keep new taps farther ahead of the death point instead of
                // repeatedly placing them only a short distance before impact.
                const float learnedLead = m_fields->ship
                    ? std::max(m_fields->leadMean, SHIP_MIN_TAP_LEAD)
                    : m_fields->leadMean;
                const float leadJitter = m_fields->ship ? 8.f : 4.f;
                j = d - static_cast<int>(
                    std::lround(learnedLead + leadJitter * N(rng))
                );
                j = std::min(std::max(j, 0), latest);
            }

            int w = std::max(2, static_cast<int>(
                std::lround(m_fields->durMean + 3.f * N(rng))
            ));

            if (static_cast<int>(p.size()) < j + w) {
                p.resize(static_cast<size_t>(j + w), p.empty() ? 0 : p.back());
            }

            for (int c = j; c < j + w; ++c) {
                p[static_cast<size_t>(c)] ^= 1;
            }

            m_fields->curJ = j;
            m_fields->curW = w;
            return p;
        };

        m_fields->curJ = -1;
        m_fields->curW = 0;
        bool ok = false;
        switch (op) {
            case OP_EARLY:  { int i = pickEdge(-1); if (i >= 0) ok = shiftEdge(i, -stepOf(true)); break; }
            case OP_LATE:   { int i = pickEdge(-1); if (i >= 0) ok = shiftEdge(i, stepOf(false)); break; }
            case OP_EXTEND: { int i = pickEdge(1);  if (i >= 0) ok = shiftEdge(i, stepOf(false)); break; }
            case OP_SHRINK: { int i = pickEdge(1);  if (i >= 0) ok = shiftEdge(i, -stepOf(true)); break; }
            case OP_REMOVE: {
                int i = pickEdge(0);
                if (i >= 0) {
                    auto it = e.erase(e.begin() + i);          // the press...
                    if (it != e.end()) e.erase(it);            // ...and its release
                    ok = true;
                }
                break;
            }
            case OP_RANDOM: {
                float x = U(rng);
                float prob = m_fields->fly ? 0.05f + 0.40f * x : 0.01f + 0.12f * x;
                int lo = U(rng) < 0.25f ? 0 : std::max(0, d - 40);   // sometimes mutate the whole pattern
                std::bernoulli_distribution flip(prob);
                bool cur = false;
                for (int c = lo; c < len; ++c) {
                    if (flip(rng)) cur = !cur;
                    if (cur) p[static_cast<size_t>(c)] ^= 1;
                }
                return p;
            }
            default: break;
        }
        if (ok) return fromEdges(e, p.size());
        return addTap();   // OP_ADD, or a move that wasn't possible
    }

    std::vector<uint8_t> candidate() {
        m_fields->curOp = -1;
        m_fields->curParentFit = -1;
        m_fields->curSweep = false;
        if (m_fields->tries == 0) return basePattern(m_fields->stack.back().tail);

        // stuck: sweep every edge of the best attempt before going back to random mutation
        // Keep the systematic sweep, but interleave a mutation every third
        // attempt so ship/wave pulse edits are not starved behind a long sweep.
        if (m_fields->tries >= SWEEP_START && !m_fields->archive.empty() &&
            (m_fields->tries % 3 != 0)) {
            if (!m_fields->sweepBuilt) buildSweep();
            while (m_fields->sweepPos < m_fields->sweep.size()) {
                auto child = applySweep(m_fields->archive[0], m_fields->sweep[m_fields->sweepPos++]);
                if (m_fields->tried.insert(child).second) {
                    m_fields->curParentFit = m_fields->archive[0].fit;
                    m_fields->curSweep = true;
                    return child;
                }
            }
        }

        Ind par;
        if (m_fields->archive.empty()) {
            par.pat = basePattern(m_fields->stack.back().tail);
            par.fit = HORIZON;
        } else {
            std::uniform_real_distribution<float> U(0.f, 1.f);
            size_t idx = 0;
            if (m_fields->archive.size() > 1 && U(m_fields->rng) >= 0.6f) {
                idx = 1 + m_fields->rng() % (m_fields->archive.size() - 1);
            }
            par = m_fields->archive[idx];
        }
        m_fields->curParentFit = par.fit;

        std::uniform_real_distribution<float> U(0.f, 1.f);
        std::vector<uint8_t> child;
        int op = OP_ADD;
        for (int attempt = 0; attempt < 8; ++attempt) {
            op = pickOp(par);
            child = mutate(par, op);
            if (U(m_fields->rng) < 0.1f) {   // sometimes stack a second change to escape local optima
                Ind tmp;
                tmp.pat = child;
                tmp.fit = par.fit;
                tmp.vdir = par.vdir;
                child = mutate(tmp, pickOp(tmp));
            }
            if (m_fields->tried.insert(child).second) break;   // skip patterns already tried
        }
        m_fields->curOp = op;
        return child;
    }

    void rewind() {
        m_fields->needRewind = false;
        if (m_fields->holding) pressButton(false);
        m_fields->holding = false;
        m_checkpointArray->removeAllObjects();
        auto& top = m_fields->stack.back();
        if (top.cp) m_checkpointArray->addObject(top.cp);
        m_fields->allowReset = true;
        this->resetLevel();
        m_fields->allowReset = false;
        if (top.holdAtCp) { pressButton(true); m_fields->holding = true; }   // flying modes only
        updateFly();
        m_fields->currentModeKey = modeSignature(m_player1);
        m_fields->modeKeyInitialized = true;
        m_fields->sectionBoundaryPending = false;
    }

    void newTrial() {
        updateFly();
        m_fields->currentModeKey = modeSignature(m_player1);
        m_fields->modeKeyInitialized = true;
        m_fields->sectionBoundaryPending = false;
        m_fields->finalSectionPending = false;
        auto& top = m_fields->stack.back();
        m_fields->committed.resize(top.len);
        m_fields->committedX.resize(top.len);
        trimCommittedTrace(top.traceLen);

        m_fields->pat = candidate();
        // Do not merge cube jumps across an entire pattern: a layer can enter
        // ship mode later, where those releases are essential flight controls.
        m_fields->patX.clear();
        m_fields->patTrace.clear();
        m_fields->candTraceCount = 0;
        m_fields->chunk = 0;
        m_fields->tickIn = 0;
    }

    // an attempt failed (death, or it reproduced a banned dead-end pattern)
    void fail() {
        clearCand();   // a checkpoint from a failed run is no longer valid
        if (++m_fields->tries >= LAYER_TRIES) {
            m_fields->tries = 0;
            resetLayerState();
            const size_t sectionFloor = std::min(
                m_fields->stack.size(), m_fields->sectionStartDepth + 1
            );
            if (m_fields->stack.size() > sectionFloor) {
                while (m_fields->stack.size() > sectionFloor) {
                    Layer top = m_fields->stack.back();
                    m_fields->stack.pop_back();
                    auto& prev = m_fields->stack.back();
                    auto b = m_fields->committed.begin() + static_cast<long>(prev.len);
                    auto e = m_fields->committed.begin() + static_cast<long>(top.len);
                    prev.banned.emplace_back(b, e);                 // this lock-in dead-ends
                    m_fields->badStarts.push_back(static_cast<int>(top.len));
                    if (top.cp) top.cp->release();
                    if (++prev.strikes < MAX_STRIKES) break;        // else drop this layer too
                }
                // The surviving checkpoint's old tail was the seed route to
                // the child we just discarded. Do not reuse it on the new trial.
                clearRegressionSeed("layer exhaustion");
                log::info("Autoplayer: dead end, backtracked to depth {} ({} chunks)",
                          m_fields->stack.size(), m_fields->stack.back().len);
            } else {
                clearRegressionSeed("section-root exhaustion");
                log::info("Autoplayer: exhausted attempts at the current section start; clearing its seed and retrying");
            }
        }
        m_fields->needRewind = true;
    }

    void onDeath() {
        m_fields->dead = false;
        int d = m_fields->chunk;
        int vd = m_fields->vdir;
        // ---------------------------------------------------------
        // SHIP STALL DETECTION
        // Tracks repeated deaths globally, even across checkpoint
        // regressions. This prevents the bot from getting trapped
        // repeatedly learning the same ship obstacle.
        // ---------------------------------------------------------
        const size_t sectionFloor = std::min(
            m_fields->stack.size(), m_fields->sectionStartDepth + 1
        );
        if (m_fields->ship && m_fields->stack.size() > sectionFloor) {

            const int globalDeath =
                static_cast<int>(m_fields->stack.back().len) + d;

            // Check whether this is still approximately the same obstacle.
            if (m_fields->shipStallGlobalChunk >= 0 &&
                std::abs(globalDeath - m_fields->shipStallGlobalChunk)
                    <= SHIP_STALL_TOLERANCE) {

                ++m_fields->shipStallDeaths;

            } else {
                // A genuinely different obstacle: start a fresh counter.
                m_fields->shipStallGlobalChunk = globalDeath;
                m_fields->shipStallDeaths = 1;
                m_fields->shipStallRegressions = 0;
            }

            const int requiredDeaths =
                m_fields->shipStallRegressions == 0
                    ? SHIP_STALL_DEATHS
                    : SHIP_STALL_NEXT_THRESHOLD;

            log::warn(
                "Autoplayer: ship stall at global chunk {} "
                "({}/{}, regression level {})",
                globalDeath,
                m_fields->shipStallDeaths,
                requiredDeaths,
                m_fields->shipStallRegressions
            );

            if (m_fields->shipStallDeaths >= requiredDeaths) {

                // First regression removes 2 checkpoints.
                // Subsequent regressions remove progressively more.
                const int regressCount = std::min(
                    SHIP_STALL_MAX_REGRESS,
                    2 + m_fields->shipStallRegressions
                );

                int removed = 0;

                while (removed < regressCount &&
                    m_fields->stack.size() > sectionFloor) {

                    Layer failed = std::move(m_fields->stack.back());
                    m_fields->stack.pop_back();

                    m_fields->badStarts.push_back(
                        static_cast<int>(failed.len)
                    );

                    if (failed.cp)
                        failed.cp->release();

                    ++removed;
                }

                // Discard all committed input after the new checkpoint.
                const size_t rewindTo = m_fields->stack.back().len;

                m_fields->committed.resize(rewindTo);
                m_fields->committedX.resize(rewindTo);
                trimCommittedTrace(m_fields->stack.back().traceLen);
                clearRegressionSeed("ship stall recovery");

                clearCand();

                m_fields->tries = 0;
                resetLayerState();

                m_fields->shipStallDeaths = 0;
                ++m_fields->shipStallRegressions;

                m_fields->needRewind = true;

                log::warn(
                    "Autoplayer: SHIP STALL RECOVERY! "
                    "Removed {} checkpoints, returning to depth {} "
                    "(chunk {}). Next regression will be deeper.",
                    removed,
                    m_fields->stack.size(),
                    rewindTo
                );

                return;
            }
        }
        if (m_fields->deaths++ < 40) {
            const float deathY = m_fields->patTrace.empty() ? -1.f : m_fields->patTrace.back().y;
            const float deathVy = m_fields->patTrace.empty() ? 0.f : m_fields->patTrace.back().vy;
            log::info("Autoplayer: death #{} depth={} try={} chunk={} parent={} op={} vdir={} y={:.2f} vy={:.2f} mode={}",
                      m_fields->deaths, m_fields->stack.size(), m_fields->tries, d,
                      m_fields->curParentFit,
                      m_fields->curSweep ? "sweep" : (m_fields->curOp >= 0 ? OP_NAME[m_fields->curOp] : "seed"),
                      vd, deathY, deathVy, m_fields->ship ? "ship" : (m_fields->wave ? "wave" : "other"));
        }
        // Detect repeated deaths at the same obstacle, even if the bot survives
        // many chunks after respawning from the checkpoint. The chunk index is
        // local to this checkpoint, so matching values identify the same region.
        auto& current = m_fields->stack.back();
        if (current.lastDeathChunk >= 0 &&
            std::abs(d - current.lastDeathChunk) <= SAME_SPOT_TOLERANCE) {
            ++current.sameSpotDeaths;
        } else {
            current.sameSpotDeaths = 1;
        }
        current.lastDeathChunk = d;
        log::warn("Autoplayer: repeated death at chunk {} ({}/{} at same spot, depth {})",
                  d, current.sameSpotDeaths, SAME_SPOT_LIMIT, m_fields->stack.size());

        if (current.sameSpotDeaths >= SAME_SPOT_LIMIT &&
            m_fields->stack.size() > sectionFloor) {
            Layer failed = std::move(current);
            m_fields->stack.pop_back();
            auto& previous = m_fields->stack.back();

            // Discard the committed segment that led to the repeatedly failing checkpoint.
            m_fields->committed.resize(previous.len);
            m_fields->committedX.resize(previous.len);
            trimCommittedTrace(previous.traceLen);
            clearRegressionSeed("same-spot regression");
            m_fields->badStarts.push_back(static_cast<int>(failed.len));
            if (failed.cp) failed.cp->release();

            clearCand();
            m_fields->tries = 0;
            resetLayerState();
            m_fields->needRewind = true;
            log::warn("Autoplayer: same obstacle killed {} attempts; reverting to depth {} ({} chunks)",
                      SAME_SPOT_LIMIT, m_fields->stack.size(), previous.len);
            return;
        }

        credit(d);
        m_fields->tried.insert(m_fields->pat);
        int prevBest = m_fields->archive.empty() ? -1 : m_fields->archive[0].fit;
        archiveAdd(m_fields->pat, d, vd);
        if (m_fields->archive[0].fit > prevBest + 2) {
            // progress: fresh budget for this layer
            if (m_fields->bestLogs++ < 40) {
                log::info("Autoplayer: best attempt {} -> {} chunks (steps: earlier {:.1f}, later {:.1f}; lead {:.1f}, hold {:.1f})",
                          prevBest, m_fields->archive[0].fit, m_fields->stepE, m_fields->stepL,
                          m_fields->leadMean, m_fields->durMean);
            }
            if (m_fields->curSweep) log::info("Autoplayer: the edge sweep found an improvement");
            m_fields->tries = 1;
            m_fields->sweepBuilt = false;   // the best attempt changed: rebuild the sweep if we get stuck again
            m_fields->sweep.clear();
            m_fields->sweepPos = 0;
            clearCand();
            m_fields->needRewind = true;
            return;
        }
        fail();
    }

    void onSurvived() {
        auto& top = m_fields->stack.back();
        if (!m_fields->candCp) { fail(); return; }
        size_t lockLen = static_cast<size_t>(m_fields->candG) - top.len;
        if (lockLen > m_fields->pat.size() || lockLen > m_fields->chunk) {
            log::error("Autoplayer: checkpoint input prefix is outside the executed pattern");
            fail();
            return;
        }

        std::vector<uint8_t> prefix(m_fields->pat.begin(), m_fields->pat.begin() + static_cast<long>(lockLen));
        if (std::find(top.banned.begin(), top.banned.end(), prefix) != top.banned.end()) {
            fail();
            return;
        }

        // Use the exact number of recorded updates at checkpoint creation.
        // Do not derive this from chunks*TICKS: the reset/control phase is not
        // a physics update, and checkpoint timing can straddle a chunk edge.
        const size_t lockUpdates = std::min(m_fields->candTraceCount, m_fields->patTrace.size());
        if (lockUpdates != m_fields->candTraceCount) {
            log::error("Autoplayer: checkpoint trace prefix is incomplete (saved={} available={})",
                       m_fields->candTraceCount, m_fields->patTrace.size());
        }

        credit(1 << 20);   // surviving is the best possible result for whatever produced it

        auto& c = m_fields->committed;
        c.insert(c.end(), prefix.begin(), prefix.end());

        // Commit the trace prefix that ends exactly at the checkpoint, and
        // carry the already-executed trace tail into the newly promoted layer.
        m_fields->committedTrace.insert(
            m_fields->committedTrace.end(),
            m_fields->patTrace.begin(),
            m_fields->patTrace.begin() + static_cast<long>(lockUpdates)
        );
        std::vector<TraceFrame> tailTrace(
            m_fields->patTrace.begin() + static_cast<long>(lockUpdates),
            m_fields->patTrace.end()
        );

        // patX has one world-space X sample for each executed local chunk.
        // Preserve the same prefix/tail split as the input bit pattern.
        if (m_fields->patX.size() < m_fields->chunk)
            m_fields->patX.resize(m_fields->chunk, m_player1->getPositionX());
        auto& committedX = m_fields->committedX;
        committedX.insert(committedX.end(), m_fields->patX.begin(),
                          m_fields->patX.begin() + static_cast<long>(lockLen));
        std::vector<float> tailX(m_fields->patX.begin() + static_cast<long>(lockLen),
                                 m_fields->patX.begin() + static_cast<long>(m_fields->chunk));

        std::vector<uint8_t> tail(m_fields->pat.begin() + static_cast<long>(lockLen),
                                  m_fields->pat.begin() + static_cast<long>(m_fields->chunk));
        Layer next;
        next.cp = m_fields->candCp;
        next.len = c.size();
        next.traceLen = m_fields->committedTrace.size();
        next.tail = tail;
        next.holdAtCp = m_fields->candHold;
        // Carry the known ship obstacle forward relative to the new checkpoint.
        // Once the checkpoint is beyond that obstacle, the old death location
        // is cleared so the next section can be learned normally.
        if (m_fields->ship && top.lastDeathChunk >= 0) {
            const int checkpointLocal = m_fields->candG - static_cast<int>(top.len);
            const int remainingDeath = top.lastDeathChunk - checkpointLocal;
            if (remainingDeath > 0) next.lastDeathChunk = remainingDeath;
        }
        m_fields->candCp = nullptr;
        m_fields->candG = -1;
        m_fields->candTraceCount = 0;
        m_fields->stack.push_back(std::move(next));

        // Keep running: the live run is already the new layer's first attempt (the seed, continued),
        // and it has already executed tail.size() chunks of it.
        m_fields->pat = basePattern(tail);
        // Preserve the exact successful input edges in the carried tail.
        m_fields->patX = std::move(tailX);
        m_fields->patTrace = std::move(tailTrace);
        m_fields->chunk = static_cast<int>(tail.size());
        m_fields->tries = 0;
        resetLayerState();
        updateFly();
        if (m_fields->stack.size() % 10 == 0) {
            log::info("Autoplayer: depth {} ({} chunks, {:.1f}%)", m_fields->stack.size(),
                      c.size(), this->getCurrentPercent());
        }
    }

    void searchTick() {
        if (!m_fields->practiceSet) {
            if (!m_isPracticeMode) this->togglePracticeMode(true);
            m_fields->practiceSet = true;
        }
        if (!m_fields->started) {
            if (++m_fields->warmup < WARMUP) return;
            m_fields->started = true;
            m_fields->engaged = true;
            cacheHazards();
            m_fields->stack.push_back(Layer{});
            m_fields->sectionStartChunk = 0;
            m_fields->sectionStartTrace = 0;
            m_fields->sectionStartDepth = 0;
            m_fields->currentModeKey = modeSignature(m_player1);
            m_fields->modeKeyInitialized = true;
            const double configuredSpeed = Mod::get()->getSettingValue<double>("speed");
            float speed = std::isfinite(configuredSpeed)
                ? static_cast<float>(configuredSpeed)
                : 1.f;
            if (!std::isfinite(speed) || speed <= 0.f) speed = 1.f;
            speed = std::clamp(speed, 0.1f, 8.f);
            m_fields->trainingTimeScale = speed;
            setSchedulerTimeScale(speed, "search start");
            log::info(
                "Autoplayer: search started (scheduler x{:.2f}); section validation enabled",
                speed
            );
        }

        if (m_fields->sectionValidationActive) {
            if (m_fields->sectionValidationDeathPending) {
                resumeSectionTrainingAfterValidationFailure("player died during continuous validation");
                return;
            }
            if (m_fields->sectionValidationCompletePending) {
                finishSectionValidation();
                return;
            }
            return; // validation owns the physics stream until it completes or dies
        }

        if (m_fields->dead) onDeath();

        if (m_fields->finalSectionPending && !m_fields->needRewind) {
            if (!beginSectionValidation(true)) {
                log::error("Autoplayer: final section could not be prepared for validation; continuing local search");
                m_fields->finalSectionPending = false;
            }
            return;
        }
        if (m_fields->sectionBoundaryPending && !m_fields->needRewind &&
            m_fields->tickIn == 0) {
            if (!beginSectionValidation(false)) {
                log::error("Autoplayer: section boundary could not be prepared for validation");
                m_fields->sectionBoundaryPending = false;
            }
            return;
        }

        // Rewind and choose the next chunk input here exactly as before. Trace
        // samples are captured separately by the PlayerObject::update hook, so
        // variable render-frame cadence cannot change the training trace length.
        if (m_fields->needRewind) {
            rewind();
            newTrial();
        }

        // The current layer may have crossed a gamemode portal since it began.
        updateFly();

        int base = static_cast<int>(m_fields->stack.back().len);
        while (m_fields->pat.size() <= static_cast<size_t>(m_fields->chunk)) {
            m_fields->pat.push_back(m_fields->pat.empty() ? 0 : m_fields->pat.back());   // keep the last state
        }

        // World X remains sampled once per training chunk for the legacy .pos
        // sidecar. It is not used to schedule the new replay.
        if (m_fields->tickIn == 0) {
            const size_t chunkIndex = static_cast<size_t>(m_fields->chunk);
            if (m_fields->patX.size() <= chunkIndex)
                m_fields->patX.resize(chunkIndex + 1);
            m_fields->patX[chunkIndex] = m_player1->getPositionX();
        }

        // Physics-step trace capture is handled by BotPlayerObject::update.

        bool want = m_fields->pat[static_cast<size_t>(m_fields->chunk)] != 0;
        applyInput(want, "SEARCH", base + m_fields->chunk);
        if (++m_fields->tickIn >= TICKS) {
            m_fields->tickIn = 0;
            ++m_fields->chunk;
            int g = base + m_fields->chunk;

            // Ship checkpoints are targeted relative to the last known death.
            // Other modes keep the original COMMIT / safe-state behaviour.
            const auto& layer = m_fields->stack.back();
            const int checkpointAt = m_fields->ship ? shipCheckpointTarget(layer) : COMMIT;
            if (!m_fields->candCp && m_fields->chunk >= checkpointAt) {
                bool safe = safeToCheckpoint(g);
                bool forced = m_fields->fly
                    ? m_fields->chunk >= std::max(checkpointAt, DEADLINE)
                    : m_fields->chunk >= MAX_RUN;
                if (safe || forced) {
                    if (auto* cp = this->createCheckpoint()) {
                        cp->retain();
                        m_fields->candCp = cp;
                        m_fields->candG = g;
                        m_fields->candHold = m_fields->holding;
                        m_fields->candTraceCount = m_fields->patTrace.size();
                        if (m_fields->cpLogs++ < 40) {
                            log::info("Autoplayer: checkpoint at chunk {} safe={} ground={} {:.1f}% traceUpdates={}",
                                      g, safe, m_player1->m_isOnGround, this->getCurrentPercent(),
                                      m_fields->candTraceCount);
                        }
                    }
                }
            }

            if (m_fields->candCp) {
                int promoteAt = HORIZON;
                if (m_fields->ship && layer.lastDeathChunk >= 0) {
                    const int checkpointLocal = m_fields->candG - base;
                    promoteAt = checkpointLocal + SHIP_CHECKPOINT_VERIFY;
                }
                if (m_fields->chunk >= promoteAt) onSurvived();
            }
        }
    }

    void compareReplayFrame(const TraceFrame& expected, const TraceFrame& actual) {

        const float dx = std::abs(actual.x - expected.x);
        const float dy = std::abs(actual.y - expected.y);
        const float dvx = std::abs(actual.vx - expected.vx);
        const float dvy = std::abs(actual.vy - expected.vy);
        const float drot = std::abs(
            std::remainder(actual.rotation - expected.rotation, 360.f)
        );
        const float ddt = std::abs(actual.dt - expected.dt);
        const float incomingDtError =
            std::abs(m_fields->replayIncomingDt - expected.dt);
        const bool flagsMatch = actual.flags == expected.flags;
        const bool inputMatch = actual.input == expected.input;

        const float positionError = std::max(dx, dy);
        const float velocityError = std::max(dvx, dvy);

        m_fields->maxPositionError = std::max(m_fields->maxPositionError, positionError);
        m_fields->maxVelocityError = std::max(m_fields->maxVelocityError, velocityError);

        const bool diverged =
            positionError > TRACE_POSITION_TOLERANCE ||
            velocityError > TRACE_VELOCITY_TOLERANCE ||
            drot > TRACE_ROTATION_TOLERANCE ||
            !flagsMatch || !inputMatch;

        if (diverged) {
            ++m_fields->divergenceCount;

            if (m_fields->firstDivergence < 0) {
                m_fields->firstDivergence = static_cast<int>(expected.update);
                log::error(
                    "Autoplayer: FIRST DIVERGENCE update={} "
                    "expected=({:.3f},{:.3f}) actual=({:.3f},{:.3f}) "
                    "dX={:.3f} dY={:.3f} dVX={:.3f} dVY={:.3f} "
                    "dRot={:.3f} dDt={:.6f} incomingDtError={:.6f} "
                    "expectedDt={:.6f} actualDt={:.6f} flagsMatch={} inputMatch={}",
                    expected.update,
                    expected.x, expected.y,
                    actual.x, actual.y,
                    dx, dy, dvx, dvy,
                    drot, ddt, incomingDtError, expected.dt, actual.dt, flagsMatch, inputMatch
                );
            }
        }

        // Periodic trajectory diagnostics.
        if (expected.update % 120 == 0) {
            log::info(
                "Autoplayer: TRACE update={} positionError={:.4f} velocityError={:.4f} "
                "divergences={} maxPosition={:.4f} maxVelocity={:.4f} "
                "dtError={:.6f} incomingDtError={:.6f} inputMatch={} flagsMatch={}",
                expected.update,
                positionError,
                velocityError,
                m_fields->divergenceCount,
                m_fields->maxPositionError,
                m_fields->maxVelocityError,
                ddt,
                incomingDtError,
                inputMatch,
                flagsMatch
            );
        }
    }

    // Save the route that exists at the moment the player leaves the level.
    // The active section is intentionally included even if it has not passed
    // continuous validation yet: replay mode should reproduce the exact
    // physics-step inputs collected so far (or the complete candidate section
    // if the trainer was in its validation pass).
    bool savePartialProgressForReplay() {
        if (!m_fields->started || m_fields->done ||
            (!m_fields->engaged && !m_fields->sectionValidationActive)) {
            return false;
        }

        std::vector<TraceFrame> snapshot = m_fields->committedTrace;

        if (m_fields->sectionValidationActive &&
            !m_fields->sectionExpectedFrames.empty()) {
            // Validation rewinds to the section root and holds the complete
            // candidate trace separately. Append that candidate to the already
            // committed prefix, rather than saving only the validation frames
            // that happened to execute before the level was exited.
            snapshot.insert(
                snapshot.end(),
                m_fields->sectionExpectedFrames.begin(),
                m_fields->sectionExpectedFrames.end()
            );
        } else {
            // During local training, patTrace is the live attempt tail after
            // the last promoted checkpoint. Earlier successful prefixes are in
            // committedTrace, so concatenating the two yields the current route.
            snapshot.insert(
                snapshot.end(),
                m_fields->patTrace.begin(),
                m_fields->patTrace.end()
            );
        }

        if (snapshot.empty()) {
            log::warn("Autoplayer: no physics frames available to save on level exit");
            return false;
        }

        // saveTrace() serializes committedTrace. Temporarily provide the
        // assembled snapshot, then restore the trainer's in-memory state.
        auto previousTrace = std::move(m_fields->committedTrace);
        m_fields->committedTrace = std::move(snapshot);
        const bool saved = saveTrace();
        const size_t savedFrames = m_fields->committedTrace.size();
        m_fields->committedTrace = std::move(previousTrace);

        log::info(
            "Autoplayer: partial progress {} on level exit; saved {} physics updates to {}",
            saved ? "saved" : "FAILED TO SAVE", savedFrames, traceSavePath().string()
        );
        return saved;
    }

    void finishReplay() {
        if (m_fields->holding && m_player1) pressButton(false);
        m_fields->holding = false;
        m_fields->replayFinished = true;
        restoreNormalTimeScale("replay finished");
        log::info(
            "Autoplayer: replay trace ended. physicsUpdates={} divergences={} firstDivergence={} "
            "maxPositionError={:.4f} maxVelocityError={:.4f}",
            m_fields->replayUpdate,
            m_fields->divergenceCount,
            m_fields->firstDivergence,
            m_fields->maxPositionError,
            m_fields->maxVelocityError
        );
    }

    void replayTick() {
        if (m_fields->replayDeathStop || m_fields->replayFinished) return;

        // Load once, then reset to the level start. From the next PlayerObject
        // physics update onward, BotPlayerObject::update consumes one trace frame
        // before the original update and compares it immediately afterwards.
        if (!m_fields->replayTraceLoaded) {
            if (m_fields->replayTraceLoadAttempted) return;
            m_fields->replayTraceLoadAttempted = true;
            if (!loadTrace()) {
                log::error("Autoplayer: could not open a valid physics trace: {}", traceSavePath().string());
                return;
            }

            restoreNormalTimeScale("replay start");

            log::info(
                "Autoplayer: loaded {} physics-step samples; replay scheduler=1x; "
                "trace was recorded at scheduler x{:.2f}",
                m_fields->replayTrace.size(), m_fields->replayRecordedTimeScale
            );
        }

        if (!m_fields->replayPrimed) {
            this->resetLevel();
            if (!m_player1 || m_fields->replayTrace.empty()) return;

            m_fields->replayUpdate = 0;
            m_fields->divergenceCount = 0;
            m_fields->firstDivergence = -1;
            m_fields->maxPositionError = 0.f;
            m_fields->maxVelocityError = 0.f;
            m_fields->hasLastReplayActual = false;
            m_fields->lastReplayExpected = TraceFrame{};
            m_fields->lastReplayActual = TraceFrame{};
            m_fields->replayPrimed = true;
            log::info("Autoplayer: replay armed at physics update 0 ({} total updates)",
                      m_fields->replayTrace.size());
        }
    }

public:
    bool isMainPlayerPhysicsUpdate(PlayerObject* player) const {
        return player && player == m_player1;
    }

    // Called immediately before PlayerObject::update. During replay, the exact
    // recorded button state is established at the same phase as in training.
    bool beforePlayerPhysicsUpdate(PlayerObject* player, float& dt) {
        if (!isMainPlayerPhysicsUpdate(player)) return false;

        if (m_fields->sectionValidationActive) {
            if (m_fields->sectionValidationDeathPending ||
                m_fields->sectionExpectedFrames.empty()) {
                return false;
            }

            // When the last expected frame has been consumed, keep the final
            // input stable until postUpdate handles the pending completion. This
            // also covers render frames containing more than one physics step.
            const size_t index = std::min(
                m_fields->sectionValidationIndex,
                m_fields->sectionExpectedFrames.size() - 1
            );
            const TraceFrame& expected = m_fields->sectionExpectedFrames[index];
            applyInput(expected.input != 0, "SECTION-VALIDATE", static_cast<int>(expected.update));
            if (std::isfinite(expected.dt) && expected.dt > 0.f) dt = expected.dt;
            m_fields->currentDt = dt;
            return true;
        }

        const bool replaying =
            !m_fields->engaged && !m_isPracticeMode &&
            Mod::get()->getSettingValue<bool>("run-replay") &&
            m_fields->replayTraceLoaded && m_fields->replayPrimed &&
            !m_fields->replayFinished && !m_fields->replayDeathStop;
        if (!replaying) return false;

        if (m_fields->replayUpdate >= m_fields->replayTrace.size()) {
            finishReplay();
            return false;
        }

        const TraceFrame& expected = m_fields->replayTrace[m_fields->replayUpdate];
        applyInput(expected.input != 0, "REPLAY", static_cast<int>(expected.update));

        // Record the timestep supplied by the game for diagnostics, then replay
        // the exact per-player physics timestep captured during training.
        m_fields->replayIncomingDt = dt;
        if (std::isfinite(expected.dt) && expected.dt > 0.f) {
            dt = expected.dt;
        }
        m_fields->currentDt = dt;
        return true;
    }

    // Called immediately after the matching PlayerObject::update returns.
    void afterPlayerPhysicsUpdate(PlayerObject* player, float dt, bool replayStep) {
        if (!isMainPlayerPhysicsUpdate(player)) return;
        ++m_fields->physicsStepsSincePostUpdate;

        if (replayStep && m_fields->sectionValidationActive) {
            if (m_fields->sectionValidationDeathPending) return;

            TraceFrame actual = captureTrace(dt);
            if (m_fields->sectionValidationIndex < m_fields->sectionExpectedFrames.size()) {
                const TraceFrame& expected =
                    m_fields->sectionExpectedFrames[m_fields->sectionValidationIndex];
                actual.update = expected.update;

                const float dx = std::abs(actual.x - expected.x);
                const float dy = std::abs(actual.y - expected.y);
                const float dvx = std::abs(actual.vx - expected.vx);
                const float dvy = std::abs(actual.vy - expected.vy);
                const float drot = std::abs(std::remainder(
                    actual.rotation - expected.rotation, 360.f));
                const bool flagsMatch = actual.flags == expected.flags;
                const bool inputMatch = actual.input == expected.input;
                const float posError = std::max(dx, dy);
                const float velError = std::max(dvx, dvy);

                m_fields->sectionValidationMaxPositionError = std::max(
                    m_fields->sectionValidationMaxPositionError, posError);
                m_fields->sectionValidationMaxVelocityError = std::max(
                    m_fields->sectionValidationMaxVelocityError, velError);
                m_fields->sectionValidationMaxRotationError = std::max(
                    m_fields->sectionValidationMaxRotationError, drot);

                if (!flagsMatch || !inputMatch) ++m_fields->sectionValidationMismatchCount;

                const float severity = std::max({
                    posError / SECTION_MAX_POSITION_ERROR,
                    velError / SECTION_MAX_VELOCITY_ERROR,
                    drot / SECTION_MAX_ROTATION_ERROR
                });
                const float worstSeverity = std::max({
                    m_fields->sectionValidationMaxPositionError / SECTION_MAX_POSITION_ERROR,
                    m_fields->sectionValidationMaxVelocityError / SECTION_MAX_VELOCITY_ERROR,
                    m_fields->sectionValidationMaxRotationError / SECTION_MAX_ROTATION_ERROR
                });
                if (severity >= worstSeverity) {
                    m_fields->sectionValidationWorstUpdate =
                        static_cast<int>(m_fields->sectionValidationIndex);
                    const float relativeVy = actual.vy * ((actual.flags & TRACE_UPSIDE_DOWN) ? -1.f : 1.f);
                    m_fields->sectionValidationFailureVdir =
                        relativeVy > 1.f ? 1 : (relativeVy < -1.f ? -1 : 0);
                }

                if (expected.update % 240 == 0) {
                    log::info(
                        "Autoplayer: SECTION CHECK update={} positionError={:.3f} velocityError={:.3f} rotationError={:.3f} inputMatch={} flagsMatch={}",
                        expected.update, posError, velError, drot, inputMatch, flagsMatch
                    );
                }
                ++m_fields->sectionValidationIndex;
            } else {
                // Additional fixed steps in the same render frame are retained
                // in the canonical trace. The last expected button state remains
                // active until searchTick can safely create the endpoint CP.
                actual.update = m_fields->sectionExpectedFrames.back().update +
                    static_cast<uint64_t>(m_fields->sectionActualFrames.size() -
                        m_fields->sectionExpectedFrames.size() + 1);
            }

            m_fields->sectionActualFrames.push_back(actual);
            if (m_fields->sectionValidationIndex >= m_fields->sectionExpectedFrames.size()) {
                m_fields->sectionValidationCompletePending = true;
            }
            return;
        }

        if (replayStep) {
            if (m_fields->replayFinished || m_fields->replayDeathStop ||
                m_fields->replayUpdate >= m_fields->replayTrace.size()) return;

            const TraceFrame expected = m_fields->replayTrace[m_fields->replayUpdate];
            TraceFrame actual = captureTrace(dt);
            actual.update = expected.update;
            m_fields->lastReplayExpected = expected;
            m_fields->lastReplayActual = actual;
            m_fields->hasLastReplayActual = true;

            // The final region is logged frame-by-frame. The normal TRACE log
            // samples only every 120 updates and can hide the exact point where
            // a previously matching trajectory changes.
            if (expected.update >= 8800 && expected.update <= 8880) {
                log::warn(
                    "Autoplayer: REPLAY TAIL update={} "
                    "expected=({:.3f},{:.3f}) actual=({:.3f},{:.3f}) "
                    "expectedV=({:.3f},{:.3f}) actualV=({:.3f},{:.3f}) "
                    "dX={:.3f} dY={:.3f} dVX={:.3f} dVY={:.3f} "
                    "expectedRot={:.3f} actualRot={:.3f} "
                    "expectedFlags=0x{:02X} actualFlags=0x{:02X} "
                    "expectedInput={} actualInput={} expectedDt={:.6f} actualDt={:.6f}",
                    expected.update,
                    expected.x, expected.y, actual.x, actual.y,
                    expected.vx, expected.vy, actual.vx, actual.vy,
                    std::abs(actual.x - expected.x),
                    std::abs(actual.y - expected.y),
                    std::abs(actual.vx - expected.vx),
                    std::abs(actual.vy - expected.vy),
                    expected.rotation, actual.rotation,
                    static_cast<unsigned>(expected.flags),
                    static_cast<unsigned>(actual.flags),
                    static_cast<int>(expected.input),
                    static_cast<int>(actual.input),
                    expected.dt, actual.dt
                );
            }

            compareReplayFrame(expected, actual);
            ++m_fields->replayUpdate;

            if (m_fields->replayUpdate >= m_fields->replayTrace.size()) {
                finishReplay();
            }
            return;
        }

        // Preserve the existing search algorithm. Only the trace sampling point
        // changes: one frame is appended for every real player physics update.
        if (m_fields->engaged && m_fields->started && !m_fields->done && !m_fields->dead &&
            !m_fields->sectionValidationActive) {
            // Detect the mode/gravity transition on the actual physics clock.
            updateFly();
            const uint8_t nextModeKey = modeSignature(m_player1);
            if (!m_fields->modeKeyInitialized) {
                m_fields->currentModeKey = nextModeKey;
                m_fields->modeKeyInitialized = true;
            } else if (nextModeKey != m_fields->currentModeKey) {
                m_fields->sectionBoundaryPending = true;
                log::info(
                    "Autoplayer: section boundary detected update={} oldMode=0x{:02X} newMode=0x{:02X}; will validate at next chunk boundary",
                    m_fields->physicsUpdate,
                    m_fields->currentModeKey,
                    nextModeKey
                );
                m_fields->currentModeKey = nextModeKey;
            }

            TraceFrame frame = captureTrace(dt);
            frame.update = m_fields->physicsUpdate++;
            m_fields->patTrace.push_back(frame);
        }
    }

    // ---- hooks ----
    void resetLevel() {
        if (m_fields->engaged && !m_fields->allowReset) return;  // ignore the game's own auto-restart during search

        // Synchronise the actual game button with our tracked state before reset.
        if (m_fields->holding && m_player1) {
            pressButton(false);
        }
        m_fields->holding = false;

        PlayLayer::resetLevel();
        enforceTrainingTimeScale("resetLevel");
        m_fields->rTick = 0;
        m_fields->replayUpdate = 0;
        m_fields->replayPrimed = false;
        // A game auto-restart after replay death must not silently start the
        // failed replay again. Toggle run-replay off/on to explicitly retry.
        m_fields->replayFinished = m_fields->replayDeathStop;
        m_fields->divergenceCount = 0;
        m_fields->firstDivergence = -1;
        m_fields->maxPositionError = 0.f;
        m_fields->maxVelocityError = 0.f;
        if (!m_fields->replayTraceLoaded) m_fields->replayTraceLoadAttempted = false;
    }

    void destroyPlayer(PlayerObject* player, GameObject* obj) {
        // The game's anti-cheat calls this with a fake spike at the start of every attempt.
        if (obj && static_cast<void*>(obj) == static_cast<void*>(m_anticheatSpike)) {
            PlayLayer::destroyPlayer(player, obj);
            return;
        }
        if (player == m_player1) {
            updateFly();
        }
        if (m_fields->sectionValidationActive && player && player == m_player1) {
            m_fields->sectionValidationDeathPending = true;
            m_fields->sectionValidationFailureIndex = m_fields->sectionValidationIndex;
            const float vy = static_cast<float>(player->m_yVelocity) *
                (player->m_isUpsideDown ? -1.f : 1.f);
            m_fields->sectionValidationFailureVdir =
                vy > 1.f ? 1 : (vy < -1.f ? -1 : 0);
            log::error(
                "Autoplayer: SECTION VALIDATION DEATH expectedFrame={} x={:.2f} y={:.2f} vy={:.3f} input={} hitX={:.2f} hitY={:.2f}",
                m_fields->sectionValidationIndex,
                player->getPositionX(), player->getPositionY(),
                static_cast<float>(player->m_yVelocity), m_fields->holding,
                obj ? obj->getPositionX() : -1.f,
                obj ? obj->getPositionY() : -1.f
            );
            PlayLayer::destroyPlayer(player, obj);
            return;
        }
        const bool replaying = !m_fields->engaged && !m_isPracticeMode &&
            Mod::get()->getSettingValue<bool>("run-replay");
        if (replaying && m_fields->replayTraceLoaded) {
            // The fatal PlayerObject::update does not reach afterPlayerPhysicsUpdate:
            // destroyPlayer is called from inside that update. Compare that exact
            // frame here, before releasing the button or stopping replay, otherwise
            // the trace diagnostics hide the mismatch that actually caused death.
            const bool wasHolding = m_fields->holding;
            const size_t deathIndex = m_fields->replayUpdate;

            if (player && deathIndex < m_fields->replayTrace.size()) {
                const TraceFrame& expected = m_fields->replayTrace[deathIndex];
                TraceFrame actual = captureTrace(m_fields->currentDt);
                actual.update = expected.update;

                const float dx = std::abs(actual.x - expected.x);
                const float dy = std::abs(actual.y - expected.y);
                const float dvx = std::abs(actual.vx - expected.vx);
                const float dvy = std::abs(actual.vy - expected.vy);
                const float drot = std::abs(std::remainder(
                    actual.rotation - expected.rotation, 360.f));

                log::error(
                    "Autoplayer: IN-UPDATE DEATH STATE update={} "
                    "(captured inside destroyPlayer; not a completed post-update sample) "
                    "expected=({:.3f},{:.3f}) actual=({:.3f},{:.3f}) "
                    "dX={:.3f} dY={:.3f} dVX={:.3f} dVY={:.3f} dRot={:.3f} "
                    "expectedInput={} actualInput={} expectedFlags=0x{:02X} actualFlags=0x{:02X} "
                    "expectedDt={:.6f} replayDt={:.6f} "
                    "ship={} bird={} ball={} dart={} swing={} ground={} "
                    "hitX={:.2f} hitY={:.2f}",
                    expected.update,
                    expected.x, expected.y, actual.x, actual.y,
                    dx, dy, dvx, dvy, drot,
                    static_cast<int>(expected.input), static_cast<int>(actual.input),
                    static_cast<unsigned>(expected.flags), static_cast<unsigned>(actual.flags),
                    expected.dt, m_fields->currentDt,
                    player->m_isShip, player->m_isBird, player->m_isBall,
                    player->m_isDart, player->m_isSwing, player->m_isOnGround,
                    obj ? obj->getPositionX() : -1.f,
                    obj ? obj->getPositionY() : -1.f
                );
            } else {
                log::error(
                    "Autoplayer: fatal replay frame unavailable update={} traceSize={}",
                    deathIndex, m_fields->replayTrace.size()
                );
            }

            if (m_fields->hasLastReplayActual) {
                const auto& prevExpected = m_fields->lastReplayExpected;
                const auto& prevActual = m_fields->lastReplayActual;
                log::warn(
                    "Autoplayer: PREVIOUS COMPLETED FRAME update={} "
                    "expected=({:.3f},{:.3f}) actual=({:.3f},{:.3f}) "
                    "expectedV=({:.3f},{:.3f}) actualV=({:.3f},{:.3f}) "
                    "dX={:.3f} dY={:.3f} dVX={:.3f} dVY={:.3f} "
                    "expectedRot={:.3f} actualRot={:.3f} "
                    "expectedInput={} actualInput={} expectedFlags=0x{:02X} actualFlags=0x{:02X} "
                    "expectedDt={:.6f} actualDt={:.6f}",
                    prevActual.update,
                    prevExpected.x, prevExpected.y, prevActual.x, prevActual.y,
                    prevExpected.vx, prevExpected.vy, prevActual.vx, prevActual.vy,
                    std::abs(prevActual.x - prevExpected.x),
                    std::abs(prevActual.y - prevExpected.y),
                    std::abs(prevActual.vx - prevExpected.vx),
                    std::abs(prevActual.vy - prevExpected.vy),
                    prevExpected.rotation, prevActual.rotation,
                    static_cast<int>(prevExpected.input), static_cast<int>(prevActual.input),
                    static_cast<unsigned>(prevExpected.flags), static_cast<unsigned>(prevActual.flags),
                    prevExpected.dt, prevActual.dt
                );
            }

            m_fields->replayFinished = true;
            m_fields->replayDeathStop = true;
            if (wasHolding) pressButton(false);
            m_fields->holding = false;
            restoreNormalTimeScale("replay death");
            log::warn(
                "Autoplayer: REPLAY DEATH physicsUpdate={} x={:.2f} y={:.2f} vy={:.2f} "
                "ground={} upsideDown={} holding={} percent={:.2f} hitX={:.2f} hitY={:.2f}",
                deathIndex,
                player ? player->getPositionX() : -1.f,
                player ? player->getPositionY() : -1.f,
                player ? static_cast<float>(player->m_yVelocity) : 0.f,
                player ? player->m_isOnGround : false,
                player ? player->m_isUpsideDown : false,
                wasHolding, this->getCurrentPercent(),
                obj ? obj->getPositionX() : -1.f,
                obj ? obj->getPositionY() : -1.f
            );
        }
        if (m_fields->engaged && !m_fields->dead) {
            m_fields->dead = true;
            m_fields->sectionBoundaryPending = false;
            m_fields->finalSectionPending = false;
            // which way was the player moving (relative to gravity)? tells us if the jump was too late or too early.
            float vy = static_cast<float>(player->m_yVelocity) * (player->m_isUpsideDown ? -1.f : 1.f);
            m_fields->vdir = vy > 1.f ? 1 : (vy < -1.f ? -1 : 0);
        }
        PlayLayer::destroyPlayer(player, obj);
    }

    void levelComplete() {
        // A locally trained route is provisional until the entire final section
        // has been replayed continuously from its section root.
        if (m_fields->sectionValidationActive) {
            m_fields->sectionValidationReachedLevelComplete = true;
            m_fields->sectionValidationCompletePending = true;
            return;
        }
        if (m_fields->engaged && !m_fields->done) {
            m_fields->finalSectionPending = true;
            return;
        }
        PlayLayer::levelComplete();
    }

    // PlayLayer is destroyed when leaving the level. Persist the in-progress
    // trace before the base class tears down the level/player state.
    void onExit() {
        savePartialProgressForReplay();
        restoreNormalTimeScale("level exit");
        PlayLayer::onExit();
    }

    void postUpdate(float dt) {
        m_fields->currentDt = dt;
        PlayLayer::postUpdate(dt);
        if (!m_player1) return;
        auto* mod = Mod::get();
        const bool searchEnabled =
            mod->getSettingValue<bool>("run-search") && !m_fields->done;

        const bool replayEnabled = !m_isPracticeMode && mod->getSettingValue<bool>("run-replay");
        if (replayEnabled != m_fields->replayWasEnabled) {
            m_fields->replayWasEnabled = replayEnabled;
            if (!replayEnabled) {
                m_fields->replayDeathStop = false;
                m_fields->replayTraceLoaded = false;
                m_fields->replayTraceLoadAttempted = false;
                m_fields->replayPrimed = false;
                m_fields->replayFinished = false;
                m_fields->replayTrace.clear();
            } else {
                m_fields->replayDeathStop = false;
                m_fields->replayTraceLoaded = false;
                m_fields->replayTraceLoadAttempted = false;
                m_fields->replayPrimed = false;
                m_fields->replayFinished = false;
            }
        }

        const uint32_t physicsStepsThisFrame = m_fields->physicsStepsSincePostUpdate;
        if (physicsStepsThisFrame > 0) {
            static int physicsDiagFrames = 0;
            if (++physicsDiagFrames % 240 == 0) {
                log::info(
                    "Autoplayer: last rendered frame contained {} player physics update(s)",
                    physicsStepsThisFrame
                );
            }
        }
        m_fields->physicsStepsSincePostUpdate = 0;

        static int tk = 0;
        if (++tk % 960 == 0) {
            log::info("Autoplayer: tick practice={} search={} replay={} started={} done={} depth={}",
                      m_isPracticeMode, mod->getSettingValue<bool>("run-search"),
                      mod->getSettingValue<bool>("run-replay"), m_fields->started,
                      m_fields->done, m_fields->stack.size());
        }

        static int speedDiagnostics = 0;

        if (++speedDiagnostics % 120 == 0) {
            auto* scheduler = CCDirector::get()->getScheduler();
            log::info(
                "Autoplayer: SPEED CHECK searchEnabled={} requested={:.2f} actual={:.2f} "
                "postUpdateDt={:.6f} playerPhysicsSteps={}",
                searchEnabled,
                m_fields->trainingTimeScale,
                scheduler->getTimeScale(),
                dt,
                physicsStepsThisFrame
            );
        }

        if (searchEnabled) {
            // This check repairs a scale change caused by a reset or another
            // scheduled game callback. searchTick() still owns the old search.
            enforceTrainingTimeScale("postUpdate");
            searchTick();
            return;
        }

        // Search has been switched off or finished. Never leave the entire game
        // running at the training multiplier while the user plays normally.
        restoreNormalTimeScale(m_fields->done ? "search complete" : "search disabled");

        if (replayEnabled) replayTick();
    }
};

// Geometry Dash calls PlayerObject::update for each player physics step (the
// fixed-step callback), whereas PlayLayer::postUpdate is render-frame based.
// Sampling here makes training and replay advance on the same simulation clock.
class $modify(BotPlayerObject, PlayerObject) {
    void update(float dt) {
        auto* playLayer = PlayLayer::get();
        if (playLayer && playLayer->m_player1 == this) {
            auto* bot = static_cast<BotLayer*>(playLayer);
            const bool replayStep = bot->beforePlayerPhysicsUpdate(this, dt);
            PlayerObject::update(dt);
            bot->afterPlayerPhysicsUpdate(this, dt, replayStep);
            return;
        }
        PlayerObject::update(dt);
    }
};
