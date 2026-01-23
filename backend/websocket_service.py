"""
📡 WEBSOCKET SERVICE - Real-time Stock Updates
===============================================
WebSocket server for real-time stock price streaming.
Automatically fetches and broadcasts live NSE stock data.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Set
import logging

from fastapi import WebSocket, WebSocketDisconnect
import yfinance as yf

logger = logging.getLogger(__name__)

# Store active connections and their subscriptions
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of symbols
        self.price_cache: Dict[str, dict] = {}
        self.cache_time: Dict[str, datetime] = {}
        self.is_broadcasting = False
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected. Total: {len(self.active_connections)}")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected. Total: {len(self.active_connections)}")
        
    async def subscribe(self, client_id: str, symbols: List[str]):
        """Subscribe client to symbols"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].update(symbols)
            # Send immediate price update for subscribed symbols
            await self.send_prices_to_client(client_id, symbols)
            
    async def unsubscribe(self, client_id: str, symbols: List[str]):
        """Unsubscribe client from symbols"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] -= set(symbols)
            
    def get_all_subscribed_symbols(self) -> Set[str]:
        """Get all unique symbols that any client is subscribed to"""
        all_symbols = set()
        for symbols in self.subscriptions.values():
            all_symbols.update(symbols)
        return all_symbols
    
    async def send_prices_to_client(self, client_id: str, symbols: List[str]):
        """Send prices for specific symbols to a client"""
        if client_id not in self.active_connections:
            return
            
        prices = {}
        for symbol in symbols:
            if symbol in self.price_cache:
                prices[symbol] = self.price_cache[symbol]
            else:
                # Fetch if not cached
                price_data = await self.fetch_single_price(symbol)
                if price_data:
                    prices[symbol] = price_data
                    
        if prices:
            try:
                await self.active_connections[client_id].send_json({
                    "type": "price_update",
                    "prices": prices,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                
    async def broadcast_prices(self):
        """Broadcast prices to all subscribed clients"""
        if not self.active_connections:
            return
            
        all_symbols = self.get_all_subscribed_symbols()
        if not all_symbols:
            return
            
        # Fetch prices in batches
        batch_size = 20
        symbols_list = list(all_symbols)
        
        for i in range(0, len(symbols_list), batch_size):
            batch = symbols_list[i:i+batch_size]
            prices = await self.fetch_batch_prices(batch)
            
            # Update cache
            for symbol, data in prices.items():
                self.price_cache[symbol] = data
                self.cache_time[symbol] = datetime.now()
                
        # Send to each client only their subscribed symbols
        for client_id, symbols in self.subscriptions.items():
            if not symbols:
                continue
                
            client_prices = {s: self.price_cache[s] for s in symbols if s in self.price_cache}
            if client_prices:
                try:
                    websocket = self.active_connections.get(client_id)
                    if websocket:
                        await websocket.send_json({
                            "type": "price_update",
                            "prices": client_prices,
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.error(f"Broadcast error for {client_id}: {e}")
                    
    async def fetch_single_price(self, symbol: str) -> dict:
        """Fetch price for a single symbol"""
        try:
            full_symbol = f"{symbol}.NS"
            ticker = yf.Ticker(full_symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return None
                
            current = float(data['Close'].iloc[-1])
            open_price = float(data['Open'].iloc[0])
            change = current - open_price
            change_pct = (change / open_price * 100) if open_price > 0 else 0
            
            return {
                "symbol": symbol,
                "price": round(current, 2),
                "change": round(change, 2),
                "change_percent": round(change_pct, 2),
                "open": round(open_price, 2),
                "high": round(float(data['High'].max()), 2),
                "low": round(float(data['Low'].min()), 2),
                "volume": int(data['Volume'].sum()),
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
            
    async def fetch_batch_prices(self, symbols: List[str]) -> Dict[str, dict]:
        """Fetch prices for multiple symbols"""
        results = {}
        
        # Use asyncio.gather for parallel fetching
        tasks = [self.fetch_single_price(s) for s in symbols]
        price_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, price_results):
            if isinstance(result, dict):
                results[symbol] = result
                
        return results


# Global connection manager
manager = ConnectionManager()


async def start_price_broadcast(interval: int = 5):
    """Background task to broadcast prices every interval seconds"""
    while True:
        try:
            if manager.active_connections:
                manager.is_broadcasting = True
                await manager.broadcast_prices()
                manager.is_broadcasting = False
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(interval)


async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint handler"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                symbols = data.get("symbols", [])
                await manager.subscribe(client_id, symbols)
                await websocket.send_json({
                    "type": "subscribed",
                    "symbols": symbols,
                    "message": f"Subscribed to {len(symbols)} symbols"
                })
                
            elif data.get("action") == "unsubscribe":
                symbols = data.get("symbols", [])
                await manager.unsubscribe(client_id, symbols)
                await websocket.send_json({
                    "type": "unsubscribed",
                    "symbols": symbols
                })
                
            elif data.get("action") == "get_price":
                symbol = data.get("symbol")
                if symbol:
                    price = await manager.fetch_single_price(symbol)
                    await websocket.send_json({
                        "type": "single_price",
                        "symbol": symbol,
                        "data": price
                    })
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)
