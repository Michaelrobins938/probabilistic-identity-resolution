# Identity Resolution Demo

A Next.js 14 application demonstrating probabilistic identity resolution and multi-touch attribution modeling.

## Features

- **Household Simulator**: Simulate multiple users with multiple devices
- **Live Session Stream**: Real-time view of user sessions and conversions
- **Identity Graph Visualization**: D3.js-powered graph showing user-device relationships
- **Attribution Dashboard**: Multiple attribution models (First Touch, Last Touch, Linear, Time Decay)
- **Professional Netflix-style UI**: Dark theme with red accents

## Tech Stack

- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- D3.js (visualization)
- Recharts (charts)
- Zustand (state management)
- Lucide React (icons)

## Getting Started

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

Open [http://localhost:3000](http://localhost:3000) to view the demo.

## Attribution Models

1. **First Touch**: 100% credit to the first interaction
2. **Last Touch**: 100% credit to the final interaction
3. **Linear**: Equal credit across all touchpoints
4. **Time Decay**: More credit to recent interactions

## License

MIT
