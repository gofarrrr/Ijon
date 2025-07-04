#!/usr/bin/env python3
"""
Clean Neon database - Remove all test extraction data.

This script removes all the test data we stored during development,
leaving the database clean for the next round of testing.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import psycopg2
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)


async def clean_database():
    """Clean all extraction data from Neon database."""
    print("🧹 Cleaning Neon Database - Removing All Test Data")
    print("=" * 60)
    
    try:
        # Get connection string
        conn_str = os.getenv('NEON_CONNECTION_STRING')
        if not conn_str:
            print("❌ NEON_CONNECTION_STRING not found in environment")
            return
        
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                
                # 1. First, let's see what we have
                print("\n📊 Current database contents:")
                
                tables_to_check = [
                    'documents', 'content_chunks', 'distilled_knowledge', 
                    'qa_pairs', 'agent_memories', 'agent_scratchpad'
                ]
                
                total_records = 0
                for table in tables_to_check:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        print(f"   - {table}: {count} records")
                        total_records += count
                    except Exception as e:
                        print(f"   - {table}: Error checking ({e})")
                
                if total_records == 0:
                    print("\n✅ Database is already clean!")
                    return
                
                print(f"\n📋 Total records to clean: {total_records}")
                
                # 2. Ask for confirmation
                print("\n⚠️  This will DELETE ALL DATA from the extraction tables.")
                print("   This includes:")
                print("   - All processed documents")
                print("   - All content chunks (including RAG chunks)")
                print("   - All distilled knowledge")
                print("   - All Q&A pairs")
                print("   - All agent memories and scratchpad data")
                
                response = input("\n❓ Are you sure you want to proceed? (yes/no): ").lower().strip()
                
                if response not in ['yes', 'y']:
                    print("❌ Cleaning cancelled.")
                    return
                
                # 3. Clean the database
                print("\n🗑️  Starting database cleanup...")
                
                # Order matters due to foreign key constraints
                cleanup_order = [
                    'content_chunks',      # References documents
                    'distilled_knowledge', # References documents  
                    'qa_pairs',           # References documents
                    'agent_memories',     # May reference documents
                    'agent_scratchpad',   # Temporary data
                    'documents'           # Main table, clean last
                ]
                
                deleted_counts = {}
                
                for table in cleanup_order:
                    try:
                        # Get count before deletion
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        before_count = cur.fetchone()[0]
                        
                        if before_count > 0:
                            # Delete all records
                            cur.execute(f"DELETE FROM {table}")
                            deleted_count = cur.rowcount
                            deleted_counts[table] = deleted_count
                            
                            print(f"   ✅ Cleaned {table}: {deleted_count} records deleted")
                        else:
                            print(f"   ⏭️  {table}: already empty")
                            
                    except Exception as e:
                        print(f"   ❌ Error cleaning {table}: {e}")
                        # Continue with other tables
                
                # 4. Commit all changes
                conn.commit()
                
                # 5. Verify cleanup
                print("\n🔍 Verifying cleanup...")
                all_clean = True
                
                for table in tables_to_check:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        remaining = cur.fetchone()[0]
                        if remaining > 0:
                            print(f"   ⚠️  {table}: {remaining} records remaining")
                            all_clean = False
                        else:
                            print(f"   ✅ {table}: clean")
                    except Exception as e:
                        print(f"   ❌ {table}: Error verifying ({e})")
                        all_clean = False
                
                # 6. Summary
                total_deleted = sum(deleted_counts.values())
                print(f"\n📊 Cleanup Summary:")
                print(f"   - Total records deleted: {total_deleted}")
                
                for table, count in deleted_counts.items():
                    print(f"   - {table}: {count} deleted")
                
                if all_clean:
                    print("\n🎉 Database cleanup completed successfully!")
                    print("   The database is now clean and ready for new testing.")
                else:
                    print("\n⚠️  Cleanup completed with some issues.")
                    print("   Please check the verification results above.")
                
    except Exception as e:
        logger.error(f"Error cleaning database: {e}")
        print(f"❌ Error during cleanup: {e}")


async def reset_extraction_states():
    """Also clean local extraction states."""
    print("\n🗂️  Cleaning local extraction states...")
    
    try:
        states_dir = Path("extraction_states")
        if states_dir.exists():
            state_files = list(states_dir.glob("*.json"))
            if state_files:
                print(f"   Found {len(state_files)} state files")
                
                for state_file in state_files:
                    try:
                        state_file.unlink()
                        print(f"   ✅ Deleted {state_file.name}")
                    except Exception as e:
                        print(f"   ❌ Error deleting {state_file.name}: {e}")
                
                print(f"   🎉 Cleaned {len(state_files)} extraction state files")
            else:
                print("   ✅ No state files to clean")
        else:
            print("   ✅ No extraction_states directory found")
            
    except Exception as e:
        print(f"   ❌ Error cleaning extraction states: {e}")


async def main():
    """Main cleanup function."""
    await clean_database()
    await reset_extraction_states()
    
    print("\n" + "=" * 60)
    print("🏁 Cleanup Complete!")
    print("   The system is now ready for fresh testing.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())